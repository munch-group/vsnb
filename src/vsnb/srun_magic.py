"""
Enhanced Jupyter cell magic for running code in an isolated subprocess with state transfer.

This version can pass the notebook's global scope to the subprocess and retrieve
the updated state back, making isolated cells contribute to the notebook scope.

Usage:
    %%srun
    # Your code here - behaves like a normal cell but runs in subprocess
    
    %%srun --no-state
    # Run truly isolated without state transfer
"""

import ast
import subprocess
import sys
import tempfile
import re
import pickle
import dill  # Better serialization than pickle
import json
import types
from pathlib import Path
from IPython.core.magic import Magics, magics_class, cell_magic
from IPython.core.getipython import get_ipython
import numpy as np
import pandas as pd
import shutil


def extract_imports_from_code(code):
    """Extract import statements from Python code."""
    imports = []
    
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.append(ast.unparse(node))
            elif isinstance(node, ast.ImportFrom):
                imports.append(ast.unparse(node))
    except SyntaxError:
        # Fallback to regex if AST parsing fails
        import_pattern = r'^\s*(from\s+[\w\.]+\s+import\s+.*|import\s+[\w\.,\s]+)$'
        for line in code.split('\n'):
            if re.match(import_pattern, line.strip()):
                imports.append(line.strip())
    
    return imports


def get_all_previous_imports(ipython):
    """Collect all import statements from cells executed before the current one."""
    all_imports = []
    seen_imports = set()
    
    history = ipython.history_manager
    session_num = history.session_number
    
    for _, _, code in history.get_range(session_num, start=1):
        if code:
            imports = extract_imports_from_code(code)
            for imp in imports:
                normalized = ' '.join(imp.split())
                if normalized not in seen_imports:
                    seen_imports.add(normalized)
                    all_imports.append(imp)
    
    return all_imports


def serialize_globals(globals_dict):
    """
    Serialize the globals dictionary for transfer to subprocess.
    Returns (serializable_dict, failed_items).
    """
    import warnings
    
    # Items to skip - IPython internals and non-serializable objects
    skip_patterns = [
        'In', 'Out', '_', '__', '___', '_i', '_ii', '_iii',
        '_oh', '_dh', 'exit', 'quit', 'get_ipython',
        '__builtins__', '__builtin__', '__package__',
        '__loader__', '__spec__', '__annotations__',
        '__cached__', '__file__', '_ih', '_oh', '_dh'
    ]
    
    serializable = {}
    failed = {}
    
    for key, value in globals_dict.items():
        # Skip IPython internals and private variables
        if key in skip_patterns or key.startswith('_i'):
            continue
            
        # Skip modules, functions from modules, and built-in functions
        if isinstance(value, types.ModuleType):
            continue
        if isinstance(value, types.BuiltinFunctionType):
            continue
        if isinstance(value, type):
            # Skip classes unless they're user-defined
            if value.__module__ not in ['__main__', '__console__']:
                continue
        
        try:
            # Try to serialize with dill (handles more types than pickle)
            serialized = dill.dumps(value)
            # Test deserialization
            dill.loads(serialized)
            serializable[key] = value
        except Exception as e:
            # If dill fails, try to handle specific types
            try:
                if isinstance(value, (np.ndarray, pd.DataFrame, pd.Series)):
                    # These should work with dill, but have a fallback
                    serializable[key] = value
                elif isinstance(value, (int, float, str, bool, list, dict, tuple, set)):
                    # Basic types should always work
                    serializable[key] = value
                else:
                    failed[key] = f"{type(value).__name__}: {str(e)[:50]}"
            except Exception as e2:
                failed[key] = f"{type(value).__name__}: {str(e2)[:50]}"
    
    return serializable, failed


def create_subprocess_script(imports, cell_code, transfer_state=True):
    """Create the Python script that will run in the subprocess."""
    
    script = '''
import sys
import dill
import pickle
import base64
import traceback

# Function to load state
def load_state():
    if len(sys.argv) > 1:
        state_file = sys.argv[1]
        try:
            with open(state_file, 'rb') as f:
                return dill.load(f)
        except Exception as e:
            print(f"Warning: Could not load state: {e}", file=sys.stderr)
            return {}
    return {}

# Function to save state
def save_state(state_dict, output_file):
    try:
        # Filter out non-serializable items for return
        clean_state = {}
        for key, value in state_dict.items():
            # Skip built-ins and internals
            if key.startswith('__') and key.endswith('__'):
                continue
            if key in ['dill', 'pickle', 'sys', 'base64', 'traceback', 
                      'load_state', 'save_state', '_initial_state']:
                continue
            try:
                dill.dumps(value)
                clean_state[key] = value
            except:
                pass  # Skip non-serializable items
        
        with open(output_file, 'wb') as f:
            dill.dump(clean_state, f)
    except Exception as e:
        print(f"Warning: Could not save state: {e}", file=sys.stderr)

'''

    if transfer_state:
        script += '''
# Load initial state
_initial_state = load_state()
globals().update(_initial_state)

'''

    # Add imports
    if imports:
        script += "# Collected imports from previous cells\n"
        script += '\n'.join(imports)
        script += '\n\n'
    
    # Add the cell code
    script += "# Cell code\n"
    script += cell_code
    script += '\n\n'
    
    if transfer_state:
        script += '''
# Save the updated state
if len(sys.argv) > 2:
    # Get the current globals, excluding the initial state variable
    current_globals = {k: v for k, v in globals().items() if k != '_initial_state'}
    save_state(current_globals, sys.argv[2])
'''
    
    return script


@magics_class
class RunSrunMagic(Magics):
    
    @cell_magic
    def srun(self, line, cell):
        """
        Run code in a subprocess with optional state transfer.
        
        Usage:
            %%srun [options]
            code...
        
        Options:
            --no-imports : Don't include previous imports
            --no-state : Don't transfer global state (true isolation)
            --python PATH : Use specific Python interpreter
            --timeout SECONDS : Set subprocess timeout (default: 30)
            --show-script : Print the generated script before running
            --show-state : Show what variables are being transferred
        """
        
        # Parse arguments
        args = line.strip().split() if line else []
        no_imports = '--no-imports' in args
        no_state = '--no-state' in args
        show_script = '--show-script' in args
        show_state = '--show-state' in args
        timeout = 30
        python_path = sys.executable
        
        # Parse timeout
        if '--timeout' in args:
            idx = args.index('--timeout')
            if idx + 1 < len(args):
                try:
                    timeout = float(args[idx + 1])
                except ValueError:
                    print(f"Warning: Invalid timeout value, using default: {timeout}")
        
        # Parse Python path
        if '--python' in args:
            idx = args.index('--python')
            if idx + 1 < len(args):
                python_path = args[idx + 1]

        # slurm memory per cpu
        if '--mem' in args:
            idx = args.index('--mem')
            if idx + 1 < len(args):
                mem = args[idx + 1]

        # slurm nr cpus
        if '--cores' in args:
            idx = args.index('--cores')
            if idx + 1 < len(args):
                cores = args[idx + 1]

        if '--time' in args:
            idx = args.index('--time')
            if idx + 1 < len(args):
                walltime = args[idx + 1]

        if '--account' in args:
            idx = args.index('--account')
            if idx + 1 < len(args):
                account = args[idx + 1]                

        # Install dill if not available
        import dill

        # Collect imports
        imports = []
        if not no_imports:
            try:
                imports = get_all_previous_imports(self.shell)
                if imports:
                    print(f"Collected {len(imports)} import statements")
            except Exception as e:
                print(f"Warning: Could not collect imports: {e}")
        
        # Prepare state transfer
        state_file = None
        output_state_file = None
        
        if not no_state:
            # Get current globals
            user_globals = self.shell.user_ns
            serializable, failed = serialize_globals(user_globals)
            
            if show_state:
                print("=" * 50)
                print("Transferring variables:")
                for key in sorted(serializable.keys()):
                    var_type = type(serializable[key]).__name__
                    print(f"  {key}: {var_type}")
                if failed:
                    print("\nNot transferring (non-serializable):")
                    for key, reason in failed.items():
                        print(f"  {key}: {reason}")
                print("=" * 50)
                print()
            else:
                print(f"Transferring {len(serializable)} variables to subprocess")
                if failed:
                    print(f"Skipping {len(failed)} non-serializable variables")
            
            # Create temporary files for state transfer
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
                dill.dump(serializable, f)
                state_file = f.name
            
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
                output_state_file = f.name
        
        # Create the script
        script = create_subprocess_script(imports, cell, transfer_state=not no_state)

        if show_script:
            print("=" * 50)
            print("Generated script:")
            print("=" * 50)
            print(script)
            print("=" * 50)
            print()
        
        # Write script to temporary file
        # with open('tmp.py', 'w') as f:
        #     f.write(script)
        #     script_file = 'tmp.py'
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            script_file = f.name
        
        try:
            # Prepare subprocess arguments
            cmd = ['srun', f'--mem={mem}', f'--cores={cores}', '--nodes=1', f'--time={walltime}', f'--account={account}']
            cmd.extend([python_path, script_file])
            if not no_state:
                cmd.extend([state_file, output_state_file])
            
            # Run the subprocess
            print(f"Running in subprocess (timeout: {timeout}s)...")
            print("-" * 50)
            
            print(" ".join(cmd))

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            # Print output
            if result.stdout:
                print(result.stdout, end='')
            if result.stderr:
                print(result.stderr, end='', file=sys.stderr)
            
            print("-" * 50)
            
            if result.returncode != 0:
                print(f"Subprocess exited with code: {result.returncode}")
            else:
                print("Subprocess completed successfully")
                
                # Load the updated state back into the notebook
                if not no_state and output_state_file:
                    try:
                        with open(output_state_file, 'rb') as f:
                            updated_state = dill.load(f)
                        
                        # Update the notebook's globals with the new state
                        updated_count = 0
                        new_vars = []
                        for key, value in updated_state.items():
                            if key not in self.shell.user_ns or \
                               not self._compare_values(self.shell.user_ns.get(key), value):
                                self.shell.user_ns[key] = value
                                if key not in serializable:
                                    new_vars.append(key)
                                else:
                                    updated_count += 1
                        
                        if updated_count > 0 or new_vars:
                            print(f"Updated {updated_count} variables in notebook scope")
                            if new_vars:
                                print(f"New variables: {', '.join(new_vars)}")
                    except Exception as e:
                        print(f"Warning: Could not load returned state: {e}")
                        
        except subprocess.TimeoutExpired:
            print(f"\nError: Subprocess timed out after {timeout} seconds")
        except Exception as e:
            print(f"\nError running subprocess: {e}")
        finally:
            # Clean up temporary files
            for temp_file in [script_file, state_file, output_state_file]:
                if temp_file:
                    Path(temp_file).unlink(missing_ok=True)
    
    def _compare_values(self, val1, val2):
        """Compare two values for equality, handling special cases."""
        try:
            if type(val1) != type(val2):
                return False
            if isinstance(val1, (np.ndarray, pd.DataFrame, pd.Series)):
                # Use appropriate comparison for these types
                if isinstance(val1, np.ndarray):
                    return np.array_equal(val1, val2)
                else:
                    return val1.equals(val2)
            else:
                return val1 == val2
        except:
            return False


# def load_ipython_extension(ipython):
#     """Load the extension in IPython."""
#     ipython.register_magics(RunSrunMagic)
#     print("Enhanced srun magic loaded. Use %%srun to run cells with state transfer.")


# def unload_ipython_extension(ipython):
#     """Unload the extension."""
#     print("Enhanced srun magic unloaded.")


# def register_magic():
#     """Manually register the magic if not loading as extension."""
#     ip = get_ipython()
#     if ip:
#         ip.register_magics(RunSrunMagic)
#         print("Enhanced srun magic registered. Use %%srun for subprocess execution with state transfer.")
#     else:
#         print("No IPython instance found. This must be run in a Jupyter environment.")


# if __name__ == "__main__":
#     print(__doc__)
