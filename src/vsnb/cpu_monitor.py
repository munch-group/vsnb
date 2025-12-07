"""
CPU Monitoring for PtDAlgorithms

This module provides live CPU usage monitoring with beautiful visualizations
for both terminal and Jupyter notebook environments. It supports:
- Per-core CPU monitoring
- SLURM multi-node awareness
- Adaptive width layouts
- Unicode bar charts in terminal
- HTML widgets in Jupyter
- Cell magic (%%monitor) for easy Jupyter usage

Usage:
    >>> # Jupyter notebook
    >>> %%monitor
    >>> result = heavy_computation()

    >>> # Python script or notebook
    >>> with pta.CPUMonitor():
    >>>     result = heavy_computation()

    >>> # Decorator
    >>> @pta.monitor_cpu
    >>> def my_function():
    >>>     pass

Author: PtDAlgorithms Team
Date: 2025-10-08
"""

import os
import time
import threading
import shutil
import functools
import subprocess
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
import re

# Core dependencies
import psutil

# Rich for terminal UI
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from rich.layout import Layout
from rich.text import Text

from IPython import get_ipython

# tqdm for Jupyter
from tqdm import tqdm as std_tqdm
try:
    from tqdm.notebook import tqdm as notebook_tqdm
    HAS_NOTEBOOK_TQDM = True
except ImportError:
    HAS_NOTEBOOK_TQDM = False
    notebook_tqdm = std_tqdm


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class NodeInfo:
    """Information about a compute node."""
    name: str
    cpu_count: int
    allocated_cpus: Optional[List[int]] = None
    process_id: int = 0
    is_local: bool = True  # Whether this is the local node
    allocated_memory_mb: Optional[float] = None  # Total allocated memory in MB

    def __post_init__(self):
        if self.allocated_cpus is None:
            self.allocated_cpus = list(range(self.cpu_count))


@dataclass
class TaskInfo:
    """Information about a SLURM task."""
    task_id: int
    node_name: str
    cpu_count: int
    allocated_cpus: Optional[List[int]] = None
    is_local: bool = True  # Whether this task runs on the local node
    allocated_memory_mb: Optional[float] = None  # Total allocated memory in MB

    def __post_init__(self):
        if self.allocated_cpus is None:
            self.allocated_cpus = list(range(self.cpu_count))

    @property
    def name(self) -> str:
        """Return a display name for this task."""
        return f"task-{self.task_id}"


@dataclass
class CPUStats:
    """Statistics for CPU usage over time."""
    samples: List[List[float]] = field(default_factory=list)  # List of per-core samples
    memory_samples: List[float] = field(default_factory=list)  # Memory usage samples
    timestamps: List[float] = field(default_factory=list)

    def add_sample(self, per_core_usage: List[float], memory_percent: float = 0.0):
        """Add a CPU usage sample."""
        self.samples.append(per_core_usage)
        self.memory_samples.append(memory_percent)
        self.timestamps.append(time.time())

    def get_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics."""
        if not self.samples:
            return {}

        import numpy as np
        samples_array = np.array(self.samples)

        duration = self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 1 else 0

        summary = {
            'duration': duration,
            'mean_per_core': np.mean(samples_array, axis=0).tolist(),
            'max_per_core': np.max(samples_array, axis=0).tolist(),
            'min_per_core': np.min(samples_array, axis=0).tolist(),
            'overall_mean': np.mean(samples_array),
            'overall_max': np.max(samples_array),
            'overall_min': np.min(samples_array),
            'cpu_seconds': np.sum(np.mean(samples_array, axis=0)) * duration / 100.0,
        }

        # Add memory statistics if available
        if self.memory_samples:
            memory_array = np.array(self.memory_samples)
            summary['memory_mean'] = np.mean(memory_array)
            summary['memory_max'] = np.max(memory_array)
            summary['memory_min'] = np.min(memory_array)

        return summary


# ============================================================================
# Environment Detection
# ============================================================================

def _clean_hostname(hostname: str) -> str:
    """
    Remove non-informative suffixes from hostnames.

    Strips common suffixes like .lan, .local, .localdomain, etc.
    """
    suffixes = ['.lan', '.local', '.localdomain', '.domain', '.home']
    for suffix in suffixes:
        if hostname.endswith(suffix):
            hostname = hostname[:-len(suffix)]
    return hostname

def _parse_slurm_cpu_list(cpu_str: str) -> List[int]:
    """
    Parse SLURM CPU list format (e.g., "0-3,8,10-11" -> [0,1,2,3,8,10,11]).

    Parameters
    ----------
    cpu_str : str
        SLURM CPU list string

    Returns
    -------
    List[int]
        List of CPU IDs
    """
    cpus = []
    for part in cpu_str.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            cpus.extend(range(start, end + 1))
        else:
            cpus.append(int(part))
    return cpus


def _query_slurm_jobs_on_node() -> Optional[Dict[str, Any]]:
    """
    Query SLURM for jobs running on the current node.

    This is useful when running outside SLURM context (e.g., VSCode on SLURM node).

    Returns
    -------
    dict or None
        Job information including job_id, nodelist, allocated CPUs, etc.
    """
    import socket
    import getpass

    try:
        current_hostname = socket.gethostname().split('.')[0]
        current_user = getpass.getuser()

        # Query squeue for jobs on this node by this user
        # Use explicit field widths to avoid truncation of nodelist
        result = subprocess.run(
            ['squeue', '-u', current_user, '-w', current_hostname,
             '-h', '-O', 'jobid:15,nodelist:100,numcpus:8,numnodes:6'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0 or not result.stdout.strip():
            return None

        # Parse first job (most recent)
        lines = result.stdout.strip().split('\n')
        if not lines:
            return None

        fields = lines[0].split()
        if len(fields) < 4:
            return None

        job_id = fields[0]
        nodelist = fields[1]
        num_cpus = int(fields[2])
        num_nodes = int(fields[3])

        # Get detailed job info including CPU IDs
        result = subprocess.run(
            ['scontrol', 'show', 'job', job_id],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0:
            return {'job_id': job_id, 'nodelist': nodelist,
                   'num_cpus': num_cpus, 'num_nodes': num_nodes}

        # Parse scontrol output for CPU_IDs
        job_info = {'job_id': job_id, 'nodelist': nodelist,
                   'num_cpus': num_cpus, 'num_nodes': num_nodes}

        for line in result.stdout.split('\n'):
            # Look for CPU_IDs on current node
            if 'CPU_IDs=' in line or 'Nodes=' in line:
                # Extract CPU list if available
                if 'CPU_IDs=' in line:
                    cpu_part = line.split('CPU_IDs=')[1].split()[0]
                    job_info['cpu_ids'] = _parse_slurm_cpu_list(cpu_part)

        return job_info

    except (subprocess.CalledProcessError, FileNotFoundError,
            subprocess.TimeoutExpired, Exception) as e:
        return None


def detect_slurm_environment() -> Dict[str, Any]:
    """
    Detect and parse SLURM environment variables.

    Returns
    -------
    dict
        Dictionary with SLURM configuration:
        - 'is_slurm': bool - Whether running under SLURM
        - 'job_id': str - Job ID
        - 'process_id': int - Process rank (SLURM_PROCID)
        - 'num_processes': int - Total processes (SLURM_NTASKS)
        - 'cpus_per_task': int - CPUs per task
        - 'nodelist': str - List of nodes
        - 'node_count': int - Number of nodes
    """
    env = {}

    # Check if running under SLURM
    env['is_slurm'] = 'SLURM_JOB_ID' in os.environ

    if not env['is_slurm']:
        # logger.info("Not running under SLURM - using single-node setup")
        return env

    # Parse SLURM environment variables
    env['job_id'] = os.environ.get('SLURM_JOB_ID')
    env['process_id'] = int(os.environ.get('SLURM_PROCID', 0))
    env['num_processes'] = int(os.environ.get('SLURM_NTASKS', 1))
    env['cpus_per_task'] = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    env['nodelist'] = os.environ.get('SLURM_JOB_NODELIST', '')
    env['node_count'] = int(os.environ.get('SLURM_JOB_NUM_NODES', 1))

    # Check if actually running under srun/sbatch (not just in allocation)
    # SLURM_STEP_ID is only set when running as part of a job step
    env['in_job_step'] = 'SLURM_STEP_ID' in os.environ

    if env['num_processes'] > 1 and not env['in_job_step']:
        # logger.warning(
        #     f"SLURM allocation has {env['num_processes']} tasks but not running under srun/sbatch.\n"
        #     f"  Distributed mode will NOT be initialized - only single-process mode.\n"
        #     f"  To enable distributed mode, run with: srun python <script>"
        # )
        # Override num_processes to prevent distributed initialization
        env['num_processes'] = 1

    # logger.info(f"SLURM environment detected:")
    # logger.info(f"  Job ID: {env['job_id']}")
    # logger.info(f"  Process: {env['process_id']}/{env['num_processes']}")
    # logger.info(f"  CPUs per task: {env['cpus_per_task']}")
    # logger.info(f"  Nodes: {env['node_count']}")
    # logger.info(f"  In job step: {env['in_job_step']}")

    return env



def detect_compute_nodes() -> List[NodeInfo]:
    """
    Detect compute nodes and allocated CPUs.

    Returns list of NodeInfo objects, one per node.
    For local execution, returns single node.
    For SLURM, returns all allocated nodes.
    """
    import socket

    # Check for SLURM environment variables
    if 'SLURM_JOB_ID' in os.environ:
        # Running inside SLURM allocation
        slurm_env = detect_slurm_environment()

        if slurm_env.get('is_slurm', False):
            # Get node list
            nodelist = slurm_env.get('nodelist', '')
            node_count = slurm_env.get('node_count', 1)
            cpus_per_task = slurm_env.get('cpus_per_task', 1)
            process_id = slurm_env.get('process_id', 0)
            job_id = slurm_env.get('job_id')

            # Parse nodelist using scontrol
            nodes = []
            if nodelist:
                try:
                    result = subprocess.run(
                        ['scontrol', 'show', 'hostnames', nodelist],
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=5
                    )
                    nodes = result.stdout.strip().split('\n')
                except (subprocess.CalledProcessError, FileNotFoundError,
                       subprocess.TimeoutExpired) as e:
                    # logger.warning(f"Could not parse SLURM nodelist: {e}")
                    nodes = [f"node-{i}" for i in range(node_count)]
            else:
                nodes = [f"node-{i}" for i in range(node_count)]

            # Get allocated CPUs from SLURM
            allocated_cpus = None
            if 'SLURM_JOB_CPUS_PER_NODE' in os.environ:
                cpu_spec = os.environ['SLURM_JOB_CPUS_PER_NODE']
                # Parse format like "4" or "4(x2)" or "4,8"
                if '(' in cpu_spec:
                    # Format: "4(x2)" means 4 CPUs per node
                    num_cpus = int(cpu_spec.split('(')[0])
                    allocated_cpus = list(range(num_cpus))
                elif ',' in cpu_spec:
                    # Multiple nodes with different CPU counts
                    num_cpus = int(cpu_spec.split(',')[process_id])
                    allocated_cpus = list(range(num_cpus))
                else:
                    num_cpus = int(cpu_spec)
                    allocated_cpus = list(range(num_cpus))

            # Try CPU affinity as fallback
            if not allocated_cpus:
                try:
                    allocated_cpus = psutil.Process().cpu_affinity()
                    if not allocated_cpus:
                        allocated_cpus = list(range(cpus_per_task))
                except (AttributeError, OSError):
                    allocated_cpus = list(range(cpus_per_task))

            # Get allocated memory from SLURM
            allocated_memory_mb = None
            if 'SLURM_MEM_PER_CPU' in os.environ:
                # Memory per CPU in MB
                mem_per_cpu = float(os.environ['SLURM_MEM_PER_CPU'])
                allocated_memory_mb = mem_per_cpu * len(allocated_cpus)
            elif 'SLURM_MEM_PER_NODE' in os.environ:
                # Total memory per node in MB
                allocated_memory_mb = float(os.environ['SLURM_MEM_PER_NODE'])
            else:
                # Fallback: use total system memory / node count
                try:
                    total_mem_mb = psutil.virtual_memory().total / (1024 ** 2)
                    allocated_memory_mb = total_mem_mb / node_count
                except:
                    allocated_memory_mb = None

            # Create NodeInfo for each node
            current_hostname = _clean_hostname(socket.gethostname())
            node_infos = []
            for i, node_name in enumerate(nodes):
                clean_name = _clean_hostname(node_name)
                is_current = (clean_name == current_hostname or i == process_id)

                node_infos.append(NodeInfo(
                    name=clean_name,
                    cpu_count=len(allocated_cpus) if is_current else cpus_per_task,
                    allocated_cpus=allocated_cpus if is_current else list(range(cpus_per_task)),
                    process_id=i,
                    is_local=is_current,
                    allocated_memory_mb=allocated_memory_mb
                ))

            return node_infos

    # Not in SLURM context - check if we're on a SLURM node with active jobs
    slurm_job = _query_slurm_jobs_on_node()

    if slurm_job:
        # Found SLURM job on this node
        # logger.info(f"Detected SLURM job {slurm_job['job_id']} on this node")

        # Parse nodelist
        nodelist = slurm_job['nodelist']
        nodes = []
        try:
            result = subprocess.run(
                ['scontrol', 'show', 'hostnames', nodelist],
                capture_output=True,
                text=True,
                check=True,
                timeout=5
            )
            nodes = result.stdout.strip().split('\n')
        except Exception as e:
            logger.warning(f"Could not parse nodelist: {e}")
            # Use current hostname
            nodes = [socket.gethostname().split('.')[0]]

        # Get allocated CPUs
        if 'cpu_ids' in slurm_job:
            allocated_cpus = slurm_job['cpu_ids']
        else:
            # Use number of CPUs as range
            allocated_cpus = list(range(slurm_job['num_cpus']))

        # Get allocated memory
        allocated_memory_mb = None
        try:
            # Fallback: use total system memory
            allocated_memory_mb = psutil.virtual_memory().total / (1024 ** 2)
        except:
            allocated_memory_mb = None

        # Create NodeInfo for each node
        current_hostname = _clean_hostname(socket.gethostname())
        node_infos = []
        for i, node_name in enumerate(nodes):
            clean_name = _clean_hostname(node_name)
            is_current = (clean_name == current_hostname)

            node_infos.append(NodeInfo(
                name=clean_name,
                cpu_count=len(allocated_cpus) if is_current else slurm_job['num_cpus'],
                allocated_cpus=allocated_cpus if is_current else list(range(slurm_job['num_cpus'])),
                process_id=i,
                is_local=is_current,
                allocated_memory_mb=allocated_memory_mb
            ))

        return node_infos

    # Local execution - no SLURM
    cpu_count = os.cpu_count() or 1

    # Try to get CPU affinity if available
    try:
        allocated_cpus = psutil.Process().cpu_affinity()
        if allocated_cpus:
            cpu_count = len(allocated_cpus)
    except (AttributeError, OSError):
        allocated_cpus = list(range(cpu_count))

    # Get total system memory
    allocated_memory_mb = None
    try:
        allocated_memory_mb = psutil.virtual_memory().total / (1024 ** 2)
    except:
        allocated_memory_mb = None

    hostname = _clean_hostname(socket.gethostname())

    return [NodeInfo(
        name=hostname,
        cpu_count=cpu_count,
        allocated_cpus=allocated_cpus,
        process_id=0,
        allocated_memory_mb=allocated_memory_mb
    )]


def _parse_slurm_tasks_per_node(tasks_str: str, num_nodes: int) -> List[int]:
    """
    Parse SLURM_TASKS_PER_NODE format.

    Parameters
    ----------
    tasks_str : str
        SLURM_TASKS_PER_NODE value (e.g., "3", "3(x4)", "3,2")
    num_nodes : int
        Number of nodes

    Returns
    -------
    List[int]
        Number of tasks on each node
    """
    if not tasks_str:
        # No info - assume uniform distribution
        return [1] * num_nodes

    tasks_per_node = []

    # Handle different formats
    parts = tasks_str.split(',')
    for part in parts:
        if '(x' in part:
            # Format: "3(x4)" means 3 tasks on each of 4 nodes
            match = re.match(r'(\d+)\(x(\d+)\)', part)
            if match:
                tasks = int(match.group(1))
                count = int(match.group(2))
                tasks_per_node.extend([tasks] * count)
        else:
            # Simple number
            tasks_per_node.append(int(part))

    # Fill remaining nodes if needed
    while len(tasks_per_node) < num_nodes:
        tasks_per_node.append(tasks_per_node[-1] if tasks_per_node else 1)

    return tasks_per_node[:num_nodes]


def detect_compute_tasks() -> List[TaskInfo]:
    """
    Detect compute tasks and allocated CPUs.

    Returns list of TaskInfo objects, one per task.
    For local execution, returns single task.
    For SLURM, returns all allocated tasks across all nodes.
    """
    import socket

    # Check for SLURM environment variables
    if 'SLURM_JOB_ID' in os.environ:
        # Running inside SLURM allocation
        slurm_env = detect_slurm_environment()

        if slurm_env.get('is_slurm', False):
            # Get task information
            num_tasks = int(os.environ.get('SLURM_NTASKS', 1))
            cpus_per_task = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
            current_task_id = int(os.environ.get('SLURM_PROCID', 0))
            num_nodes = int(os.environ.get('SLURM_NNODES', 1))
            tasks_per_node_str = os.environ.get('SLURM_TASKS_PER_NODE', '')
            nodelist = os.environ.get('SLURM_JOB_NODELIST', '')

            # Parse task distribution
            tasks_per_node = _parse_slurm_tasks_per_node(tasks_per_node_str, num_nodes)

            # Parse nodelist to get node names
            nodes = []
            current_hostname = _clean_hostname(socket.gethostname())

            if nodelist:
                try:
                    result = subprocess.run(
                        ['scontrol', 'show', 'hostnames', nodelist],
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=5
                    )
                    nodes = result.stdout.strip().split('\n')
                except (subprocess.CalledProcessError, FileNotFoundError,
                       subprocess.TimeoutExpired) as e:
                    # logger.warning(f"Could not parse SLURM nodelist: {e}")
                    # Fallback: if single node, use current hostname; otherwise use generic names
                    if num_nodes == 1:
                        nodes = [current_hostname]
                    else:
                        nodes = [f"node-{i}" for i in range(num_nodes)]
            else:
                # No nodelist - use current hostname for single node, generic names otherwise
                if num_nodes == 1:
                    nodes = [current_hostname]
                else:
                    nodes = [f"node-{i}" for i in range(num_nodes)]

            # Get allocated memory per task from SLURM
            allocated_memory_mb = None
            if 'SLURM_MEM_PER_CPU' in os.environ:
                # Memory per CPU in MB
                mem_per_cpu = float(os.environ['SLURM_MEM_PER_CPU'])
                allocated_memory_mb = mem_per_cpu * cpus_per_task
            elif 'SLURM_MEM_PER_NODE' in os.environ:
                # Total memory per node - divide by tasks per node
                mem_per_node = float(os.environ['SLURM_MEM_PER_NODE'])
                # Use average tasks per node
                avg_tasks_per_node = num_tasks / num_nodes
                allocated_memory_mb = mem_per_node / avg_tasks_per_node if avg_tasks_per_node > 0 else None
            else:
                # Fallback: use total system memory / number of tasks
                try:
                    total_mem_mb = psutil.virtual_memory().total / (1024 ** 2)
                    allocated_memory_mb = total_mem_mb / num_tasks
                except:
                    allocated_memory_mb = None

            # Build task-to-node mapping
            task_infos = []
            task_id = 0

            for node_idx, (node_name, tasks_on_node) in enumerate(zip(nodes, tasks_per_node)):
                clean_node_name = _clean_hostname(node_name)
                is_current_node = (clean_node_name == current_hostname)

                for local_task_idx in range(tasks_on_node):
                    is_current_task = (task_id == current_task_id)

                    # Get CPU affinity for current task
                    if is_current_task:
                        try:
                            allocated_cpus = psutil.Process().cpu_affinity()
                            if not allocated_cpus:
                                allocated_cpus = list(range(cpus_per_task))
                        except (AttributeError, OSError):
                            allocated_cpus = list(range(cpus_per_task))
                    else:
                        # For other tasks, assume sequential CPU allocation
                        # This is an approximation - actual affinity may differ
                        cpu_start = local_task_idx * cpus_per_task
                        allocated_cpus = list(range(cpu_start, cpu_start + cpus_per_task))

                    task_infos.append(TaskInfo(
                        task_id=task_id,
                        node_name=clean_node_name,
                        cpu_count=cpus_per_task,
                        allocated_cpus=allocated_cpus,
                        is_local=is_current_node,  # All tasks on current node are local for monitoring
                        allocated_memory_mb=allocated_memory_mb
                    ))
                    task_id += 1

            return task_infos

    # Not in SLURM context - check if we're on a SLURM node with active jobs
    slurm_job = _query_slurm_jobs_on_node()

    if slurm_job:
        # Found SLURM job on this node - try to extract task info
        num_tasks = slurm_job.get('num_tasks', 1)
        cpus_per_task = slurm_job.get('num_cpus', os.cpu_count() or 1)

        hostname = _clean_hostname(socket.gethostname())

        # Get allocated memory
        allocated_memory_mb = None
        try:
            # Fallback: use total system memory / number of tasks
            total_mem_mb = psutil.virtual_memory().total / (1024 ** 2)
            allocated_memory_mb = total_mem_mb / num_tasks
        except:
            allocated_memory_mb = None

        # Create tasks (all local since they're on the current node)
        task_infos = []
        for task_id in range(num_tasks):
            cpu_start = task_id * cpus_per_task
            allocated_cpus = list(range(cpu_start, cpu_start + cpus_per_task))

            task_infos.append(TaskInfo(
                task_id=task_id,
                node_name=hostname,
                cpu_count=cpus_per_task,
                allocated_cpus=allocated_cpus,
                is_local=True,  # All tasks on this node are local for monitoring
                allocated_memory_mb=allocated_memory_mb
            ))

        return task_infos

    # Local execution - no SLURM
    cpu_count = os.cpu_count() or 1

    # Try to get CPU affinity if available
    try:
        allocated_cpus = psutil.Process().cpu_affinity()
        if allocated_cpus:
            cpu_count = len(allocated_cpus)
    except (AttributeError, OSError):
        allocated_cpus = list(range(cpu_count))

    # Get total system memory
    allocated_memory_mb = None
    try:
        allocated_memory_mb = psutil.virtual_memory().total / (1024 ** 2)
    except:
        allocated_memory_mb = None

    hostname = _clean_hostname(socket.gethostname())

    return [TaskInfo(
        task_id=0,
        node_name=hostname,
        cpu_count=cpu_count,
        allocated_cpus=allocated_cpus,
        is_local=True,
        allocated_memory_mb=allocated_memory_mb
    )]


def is_jupyter() -> bool:
    """Check if running in Jupyter notebook."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        shell_type = str(type(shell))
        return 'ZMQInteractiveShell' in shell_type
    except (ImportError, NameError):
        return False


def is_vscode() -> bool:
    """Check if running in VSCode Jupyter."""
    try:
        import os
        # VSCode sets these environment variables
        return 'VSCODE_PID' in os.environ or 'VSCODE_CWD' in os.environ
    except:
        return False


# ============================================================================
# Remote Monitoring via SSH
# ============================================================================

def _get_remote_cpu_usage(hostname: str, allocated_cpus: List[int], timeout: int = 5) -> tuple:
    """
    Get CPU usage from a remote node via SSH.

    Parameters
    ----------
    hostname : str
        Remote hostname
    allocated_cpus : List[int]
        List of CPU IDs to monitor
    timeout : int
        SSH timeout in seconds

    Returns
    -------
    tuple
        (per_core_usage, memory_used_mb) or (None, None) on error
    """
    try:
        # Build Python command to get CPU stats
        # We'll use a one-liner that imports psutil and outputs JSON
        cpu_ids_str = ','.join(map(str, allocated_cpus))
        python_cmd = (
            f"python3 -c \"import psutil, json; "
            f"percpu = psutil.cpu_percent(interval=0.1, percpu=True); "
            f"mem_mb = psutil.virtual_memory().used / (1024 ** 2); "
            f"cpus = [{cpu_ids_str}]; "
            f"usage = [percpu[i] if i < len(percpu) else 0.0 for i in cpus]; "
            f"print(json.dumps({{'cpu': usage, 'mem': mem_mb}}))\""
        )

        # Execute via SSH
        result = subprocess.run(
            ['ssh', '-o', 'StrictHostKeyChecking=no',
             '-o', 'ConnectTimeout=2',
             hostname, python_cmd],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if result.returncode != 0:
            # logger.debug(f"SSH to {hostname} failed: {result.stderr}")
            return None, None

        # Parse JSON output
        import json
        data = json.loads(result.stdout.strip())
        return data['cpu'], data['mem']

    except (subprocess.TimeoutExpired, subprocess.CalledProcessError,
            json.JSONDecodeError, Exception) as e:
        # logger.debug(f"Could not get remote CPU usage from {hostname}: {e}")
        return None, None


# ============================================================================
# Module-level Cache
# ============================================================================

# Detect and cache compute nodes/tasks on module import
# This avoids redundant detection when creating multiple CPUMonitor instances
_CACHED_NODES = None
_CACHED_TASKS = None

def get_cached_nodes() -> List[NodeInfo]:
    """
    Get cached compute nodes, detecting them if not already cached.

    Returns
    -------
    List[NodeInfo]
        List of detected compute nodes
    """
    global _CACHED_NODES
    if _CACHED_NODES is None:
        _CACHED_NODES = detect_compute_nodes()
    return _CACHED_NODES


def get_cached_tasks() -> List[TaskInfo]:
    """
    Get cached compute tasks, detecting them if not already cached.

    Returns
    -------
    List[TaskInfo]
        List of detected compute tasks
    """
    global _CACHED_TASKS
    if _CACHED_TASKS is None:
        _CACHED_TASKS = detect_compute_tasks()
    return _CACHED_TASKS


# ============================================================================
# CPU Monitor Class
# ============================================================================

class CPUMonitor:
    """
    Live CPU monitoring with adaptive display.

    Monitors per-core CPU usage and displays it in a grid layout.
    Automatically detects SLURM nodes and Jupyter environment.

    Parameters
    ----------
    width : int, optional
        Display width in characters. If None, auto-detects terminal/notebook width.
    update_interval : float, default=0.5
        Time between updates in seconds
    show_summary : bool, default=True
        Show summary statistics when monitoring ends
    persist : bool, default=False
        If True, keep display visible after completion. If False (default),
        the display disappears when monitoring stops.
    color : bool, default=False
        If True, use color coding (green < 50%, yellow 50-80%, red > 80%).
        If False (default), all bars are gray.
    summary : bool, default=False
        If True, display results as an HTML table with mean CPU usage per core
        instead of progress bars. Memory usage (mean/max) is shown next to the
        node name. The table will be shown after completion regardless of persist setting.
    fold : int, default=16
        Maximum number of CPU bars per row before wrapping to a new line.
        Bars are automatically redistributed across rows to minimize empty
        space in the last row. Only applies to HTML display (Jupyter/VSCode).
    group_by : str, default="node"
        How to group CPU bars: "node" groups by physical node, "task" groups
        by SLURM task. For non-SLURM environments, both modes behave the same.

    Examples
    --------
    >>> with CPUMonitor():
    ...     result = computation()

    >>> with CPUMonitor(width=120, update_interval=1.0):
    ...     result = computation()

    >>> with CPUMonitor(persist=True):
    ...     result = computation()  # Display remains after completion

    >>> with CPUMonitor(color=True):
    ...     result = computation()  # Use color coding

    >>> with CPUMonitor(summary=True):
    ...     result = computation()  # Show table with CPU and memory stats
    """

    def __init__(
        self,
        width: Optional[int] = None,
        update_interval: float = 0.5,
        show_summary: bool = True,
        persist: bool = False,
        color: bool = False,
        summary: bool = False,
        fold: int = 16,
        group_by: str = "node"
    ):
        self.width = width
        self.update_interval = update_interval
        self.show_summary = show_summary
        self.persist = persist
        self.color = color
        self.summary = summary
        self.fold = fold
        self.group_by = group_by

        # Validate group_by parameter
        if group_by not in ("node", "task"):
            raise ValueError(f"group_by must be 'node' or 'task', got '{group_by}'")

        # Detect environment
        self.is_jupyter = is_jupyter()
        self.is_vscode = is_vscode()

        # Get compute units (nodes or tasks) based on grouping mode
        if group_by == "task":
            self.units = get_cached_tasks()
        else:
            self.units = get_cached_nodes()

        # Keep nodes reference for compatibility
        self.nodes = self.units

        # Monitoring state
        self._monitoring = False
        self._monitor_thread = None
        self._jupyter_update_thread = None
        self._stats = {unit.name: CPUStats() for unit in self.units}
        self._current_usage = {unit.name: [] for unit in self.units}
        self._current_memory = {unit.name: 0.0 for unit in self.units}

        # Display components
        self._console = Console()
        self._live = None
        self._tqdm_bars = []
        self._has_ipython_display = False
        self._use_html_display = False
        self._html_display = None

    def _get_display_width(self) -> int:
        """Get display width (auto-detect or user-specified)."""
        if self.width is not None:
            return self.width

        if self.is_jupyter:
            # Jupyter - use a reasonable default (will be responsive in HTML)
            return 120
        else:
            # Terminal - use terminal width
            return shutil.get_terminal_size().columns

    def _monitor_loop(self):
        """Background thread that samples CPU usage for all units (nodes or tasks)."""
        import socket
        current_hostname = _clean_hostname(socket.gethostname())

        # Identify local and remote units
        local_units = [u for u in self.units if u.is_local]
        remote_units = [u for u in self.units if not u.is_local]

        if not local_units:
            # logger.warning("Could not identify local unit")
            return

        while self._monitoring:
            try:
                # Monitor local units
                for unit in local_units:
                    # Get per-core CPU usage
                    per_core = psutil.cpu_percent(interval=self.update_interval, percpu=True)

                    # Get memory usage in MB
                    memory_used_mb = psutil.virtual_memory().used / (1024 ** 2)

                    # Filter to allocated CPUs if available
                    if unit.allocated_cpus:
                        try:
                            per_core = [per_core[i] for i in unit.allocated_cpus if i < len(per_core)]
                        except IndexError:
                            pass  # Use all cores

                    # Store current usage
                    self._current_usage[unit.name] = per_core
                    self._current_memory[unit.name] = memory_used_mb

                    # Record statistics
                    self._stats[unit.name].add_sample(per_core, memory_used_mb)

                # Monitor remote units via SSH
                for unit in remote_units:
                    # For task mode, need to SSH to the node where task runs
                    if self.group_by == "task":
                        # TaskInfo has node_name attribute
                        hostname = unit.node_name
                    else:
                        # NodeInfo's name is the hostname
                        hostname = unit.name

                    per_core, memory_used_mb = _get_remote_cpu_usage(
                        hostname,
                        unit.allocated_cpus or list(range(unit.cpu_count)),
                        timeout=3
                    )

                    if per_core is not None:
                        # Store current usage
                        self._current_usage[unit.name] = per_core
                        self._current_memory[unit.name] = memory_used_mb

                        # Record statistics
                        self._stats[unit.name].add_sample(per_core, memory_used_mb)
                    else:
                        # SSH failed - keep previous values or initialize to zeros
                        if unit.name not in self._current_usage or not self._current_usage[unit.name]:
                            self._current_usage[unit.name] = [0.0] * unit.cpu_count
                            self._current_memory[unit.name] = 0.0

            except Exception as e:
                # logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.update_interval)

    def _create_terminal_display(self) -> Table:
        """Create rich Table for terminal display."""
        display_width = self._get_display_width()

        # Calculate layout
        # Each node column needs: "CPU X [bar] XX.X%"
        # Estimate: label(6) + bar(10) + percentage(6) + padding(2) = 24 chars minimum
        min_col_width = 24
        max_nodes_per_row = max(1, display_width // min_col_width)

        # Create main table
        if len(self.nodes) == 1:
            # Single node - simple layout
            node = self.nodes[0]
            table = Table(title=f"CPU Monitor - {node.name}", show_header=False,
                         box=None, padding=(0, 1))

            usage = self._current_usage.get(node.name, [])
            for i, cpu_usage in enumerate(usage):
                # Create bar
                bar_width = 10
                filled = int((cpu_usage / 100.0) * bar_width)
                bar = '█' * filled + '░' * (bar_width - filled)

                # Color based on usage
                if cpu_usage < 50:
                    color = "green"
                elif cpu_usage < 80:
                    color = "yellow"
                else:
                    color = "red"

                table.add_row(
                    f"CPU {i:2d}",
                    Text(f"▕{bar}▏", style=color),
                    f"{cpu_usage:5.1f}%"
                )

            return table

        else:
            # Multi-node layout - create grid
            nodes_per_row = min(len(self.nodes), max_nodes_per_row)
            rows_needed = (len(self.nodes) + nodes_per_row - 1) // nodes_per_row

            # Create outer table for grid
            grid = Table(show_header=False, box=None, padding=(0, 2))
            for _ in range(nodes_per_row):
                grid.add_column()

            # Add nodes to grid
            for row_idx in range(rows_needed):
                row_nodes = []
                for col_idx in range(nodes_per_row):
                    node_idx = row_idx * nodes_per_row + col_idx
                    if node_idx >= len(self.nodes):
                        row_nodes.append("")
                        continue

                    node = self.nodes[node_idx]
                    usage = self._current_usage.get(node.name, [])

                    # Create node panel
                    node_table = Table(show_header=False, box=None, padding=(0, 0))
                    for i, cpu_usage in enumerate(usage):
                        bar_width = 8
                        filled = int((cpu_usage / 100.0) * bar_width)
                        bar = '█' * filled + '░' * (bar_width - filled)

                        if cpu_usage < 50:
                            color = "green"
                        elif cpu_usage < 80:
                            color = "yellow"
                        else:
                            color = "red"

                        node_table.add_row(
                            f"CPU {i}",
                            Text(f"▕{bar}▏", style=color),
                            f"{cpu_usage:4.1f}%"
                        )

                    panel = Panel(node_table, title=node.name, border_style="blue")
                    row_nodes.append(panel)

                grid.add_row(*row_nodes)

            return grid

    def _create_jupyter_display(self):
        """Create tqdm bars for Jupyter display."""
        # VSCode has issues with tqdm.notebook - use HTML display instead
        if self.is_vscode:
            try:
                from IPython.display import display, HTML
                self._has_ipython_display = True
                self._use_html_display = True

                # Initialize HTML display container
                for unit in self.units:
                    usage = self._current_usage.get(unit.name, [])
                    if not usage:
                        usage = [0.0] * unit.cpu_count
                        self._current_usage[unit.name] = usage

                # Create initial HTML display
                html = self._generate_html_display()
                self._html_display = display(HTML(html), display_id=True)
                return
            except:
                # Fall through to tqdm if HTML fails
                pass

        if not HAS_NOTEBOOK_TQDM:
            # Fallback to terminal-style
            return self._create_terminal_display()

        # Force display update in Jupyter/VSCode
        try:
            from IPython.display import display, clear_output
            self._has_ipython_display = True
        except ImportError:
            self._has_ipython_display = False

        self._use_html_display = False

        # Create tqdm bars for each CPU on each unit
        self._tqdm_bars = []

        for unit in self.units:
            # Get current usage or initialize to zeros
            usage = self._current_usage.get(unit.name, [])
            if not usage:
                usage = [0.0] * unit.cpu_count
                self._current_usage[unit.name] = usage

            # Unit header
            print(f"\n{unit.name} ({len(usage)} cores):")

            for i in range(len(usage)):
                bar = notebook_tqdm(
                    total=100,
                    desc=f"  CPU {i}",
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {postfix}",
                    leave=True,
                    initial=usage[i] if i < len(usage) else 0
                )
                bar.n = usage[i] if i < len(usage) else 0
                bar.refresh()
                self._tqdm_bars.append(bar)

    def _generate_html_table(self):
        """Generate HTML table with CPU statistics."""
        html = '<div style="font-family: monospace; font-size: 10px; padding: 10px;">'

        for unit_idx, unit in enumerate(self.units):
            stats = self._stats[unit.name]
            summary = stats.get_summary()
            if not summary:
                continue

            # Add separator between units (except before first)
            if unit_idx > 0:
                html += '<div style="height: 12px;"></div>'

            # Start unit container
            unit_style = 'margin-bottom: 12px;'
            html += f'<div style="{unit_style}">'

            # Unit name with memory percentage (mean/max) - convert MB to percentage
            memory_mean_mb = summary.get('memory_mean', 0.0)
            memory_max_mb = summary.get('memory_max', 0.0)
            if unit.allocated_memory_mb and unit.allocated_memory_mb > 0:
                memory_mean_pct = (memory_mean_mb / unit.allocated_memory_mb) * 100
                memory_max_pct = (memory_max_mb / unit.allocated_memory_mb) * 100
            else:
                # Fallback: show as system percentage
                total_mem_mb = psutil.virtual_memory().total / (1024 ** 2)
                memory_mean_pct = (memory_mean_mb / total_mem_mb) * 100 if total_mem_mb > 0 else 0
                memory_max_pct = (memory_max_mb / total_mem_mb) * 100 if total_mem_mb > 0 else 0
            html += f'<div style="margin-bottom: 8px; font-size: 11px;">{unit.name} <span style="font-weight: normal; color: #666;">({memory_mean_pct:.0f}%/{memory_max_pct:.0f}% mem)</span></div>'

            mean_per_core = summary['mean_per_core']
            n_cpus = len(mean_per_core)

            # Create table with single row of CPU percentages
            html += '''<table style="border-collapse: collapse; width: 100%;">'''

            # CPU usage row
            html += '<tr style="background-color: rgba(128, 128, 128, 0.1);">'
            html += '<td style="padding: 4px 8px; font-weight: bold;">CPU %</td>'
            for cpu_val in mean_per_core:
                html += f'<td style="padding: 4px 8px; text-align: center;">{cpu_val:.0f}</td>'
            html += '</tr>'

            html += '</table>'
            html += '</div>'  # Close unit container

        html += '</div>'
        return html

    def _calculate_row_distribution(self, total_cpus: int):
        """
        Calculate optimal distribution of CPUs across rows to minimize empty space.

        Parameters
        ----------
        total_cpus : int
            Total number of CPUs to distribute

        Returns
        -------
        list of int
            Number of CPUs per row
        """
        max_per_row = self.fold
        num_rows = (total_cpus + max_per_row - 1) // max_per_row  # ceiling division

        # Distribute CPUs evenly across rows
        base_cpus_per_row = total_cpus // num_rows
        extra_cpus = total_cpus % num_rows

        # First 'extra_cpus' rows get one additional CPU
        row_distribution = []
        for i in range(num_rows):
            if i < extra_cpus:
                row_distribution.append(base_cpus_per_row + 1)
            else:
                row_distribution.append(base_cpus_per_row)

        return row_distribution

    def _generate_html_display(self, summary_mode=False):
        """Generate HTML for VSCode display."""
        # If summary mode is enabled and we're in summary, show table
        if self.summary and summary_mode:
            return self._generate_html_table()

        html = '<div style="font-family: monospace; font-size: 10px; padding: 10px;">'

        for unit_idx, unit in enumerate(self.units):
            # Add separator between units (except before first)
            if unit_idx > 0:
                html += '<div style="height: 12px;"></div>'

            # Start unit container
            unit_style = 'margin-bottom: 12px;'
            html += f'<div style="{unit_style}">'

            if summary_mode:
                # Summary mode: show mean usage
                stats = self._stats[unit.name]
                summary = stats.get_summary()
                if not summary:
                    html += '</div>'  # Close unit container
                    continue

                # Unit name with mean/max memory - convert MB to percentage
                memory_mean_mb = summary.get('memory_mean', 0.0)
                memory_max_mb = summary.get('memory_max', 0.0)
                if unit.allocated_memory_mb and unit.allocated_memory_mb > 0:
                    memory_mean_pct = (memory_mean_mb / unit.allocated_memory_mb) * 100
                    memory_max_pct = (memory_max_mb / unit.allocated_memory_mb) * 100
                else:
                    # Fallback: show as system percentage
                    total_mem_mb = psutil.virtual_memory().total / (1024 ** 2)
                    memory_mean_pct = (memory_mean_mb / total_mem_mb) * 100 if total_mem_mb > 0 else 0
                    memory_max_pct = (memory_max_mb / total_mem_mb) * 100 if total_mem_mb > 0 else 0
                html += f'<div style="margin-bottom: 6px; font-size: 11px;">{unit.name} <span style="font-weight: normal; color: #666;">({memory_mean_pct:.0f}%/{memory_max_pct:.0f}% mem)</span></div>'

                n_cpus = len(summary['mean_per_core'])

                # Calculate optimal row distribution to minimize empty space
                row_distribution = self._calculate_row_distribution(n_cpus)
                max_cpus_per_row = max(row_distribution) if row_distribution else self.fold
                # Calculate gap width based on maximum CPUs per row (for consistent bar width)
                gap_width_px = (max_cpus_per_row - 1) * 3

                # Distribute CPUs across rows
                cpu_idx = 0
                for cpus_in_row in row_distribution:
                    html += '<div style="display: flex; gap: 3px; width: 100%; margin-bottom: 3px;">'
                    for i in range(cpu_idx, cpu_idx + cpus_in_row):
                        mean_val = summary['mean_per_core'][i]

                        # Color based on mode
                        summary_color = '#4CAF50' if self.color else '#666666'

                        # Show mean usage bar with fixed width using calc()
                        # Use max_cpus_per_row for consistent width across all rows
                        html += f'''
                        <div style="width: calc((100% - {gap_width_px}px) / {max_cpus_per_row}); min-width: 20px; height: 8px; background: rgba(128, 128, 128, 0.2); border-radius: 2px; overflow: hidden;" title="CPU {i}: {mean_val:.1f}% avg">
                            <div style="width: {mean_val}%; height: 100%; background: {summary_color};"></div>
                        </div>
                        '''
                    cpu_idx += cpus_in_row
                    html += '</div>'

            else:
                # Live mode: show current usage
                usage = self._current_usage.get(unit.name, [])
                if not usage:
                    usage = [0.0] * unit.cpu_count

                # Calculate memory usage percentage and get max
                current_memory_mb = self._current_memory.get(unit.name, 0.0)

                # Get max memory from stats
                stats = self._stats[unit.name]
                if stats.memory_samples:
                    max_memory_mb = max(stats.memory_samples)
                else:
                    max_memory_mb = current_memory_mb

                # Convert to percentages
                if unit.allocated_memory_mb and unit.allocated_memory_mb > 0:
                    current_mem_pct = (current_memory_mb / unit.allocated_memory_mb) * 100
                    max_mem_pct = (max_memory_mb / unit.allocated_memory_mb) * 100
                else:
                    # Fallback: show as system percentage
                    total_mem_mb = psutil.virtual_memory().total / (1024 ** 2)
                    current_mem_pct = (current_memory_mb / total_mem_mb) * 100 if total_mem_mb > 0 else 0
                    max_mem_pct = (max_memory_mb / total_mem_mb) * 100 if total_mem_mb > 0 else 0

                # Determine color based on max memory threshold
                if max_mem_pct >= 90:
                    mem_color = '#FF0000'  # red
                elif max_mem_pct >= 75:
                    mem_color = '#FF69B4'  # pink
                else:
                    mem_color = '#666666'  # gray

                # Unit name with current/max memory
                html += f'<div style="margin-bottom: 6px; font-size: 11px;">{unit.name} <span style="font-weight: normal; color: {mem_color};">({current_mem_pct:.0f}%/{max_mem_pct:.0f}% mem)</span></div>'

                n_cpus = len(usage)

                # Calculate optimal row distribution to minimize empty space
                row_distribution = self._calculate_row_distribution(n_cpus)
                max_cpus_per_row = max(row_distribution) if row_distribution else self.fold
                # Calculate gap width based on maximum CPUs per row (for consistent bar width)
                gap_width_px = (max_cpus_per_row - 1) * 3

                # Distribute CPUs across rows
                cpu_idx = 0
                for cpus_in_row in row_distribution:
                    html += '<div style="display: flex; gap: 3px; width: 100%; margin-bottom: 3px;">'
                    for i in range(cpu_idx, cpu_idx + cpus_in_row):
                        cpu_usage = usage[i]

                        # Color based on usage
                        if self.color:
                            # Color mode: green/yellow/red
                            if cpu_usage < 50:
                                color = '#4CAF50'  # green
                            elif cpu_usage < 80:
                                color = '#FFC107'  # yellow
                            else:
                                color = '#F44336'  # red
                        else:
                            # Default: gray only
                            color = '#666666'  # gray

                        # Progress bar with tooltip and fixed width using calc()
                        # Use max_cpus_per_row for consistent width across all rows
                        width_pct = min(100, max(0, cpu_usage))
                        html += f'''
                        <div style="width: calc((100% - {gap_width_px}px) / {max_cpus_per_row}); min-width: 20px; height: 8px; background: rgba(128, 128, 128, 0.2); border-radius: 2px; overflow: hidden;" title="CPU {i}: {cpu_usage:.1f}%">
                            <div style="width: {width_pct}%; height: 100%; background: {color}; transition: width 0.3s;"></div>
                        </div>
                        '''
                    cpu_idx += cpus_in_row
                    html += '</div>'

            html += '</div>'  # Close unit container

        html += '</div>'  # Close main container
        return html

    def _update_jupyter_display(self):
        """Update tqdm bars in Jupyter."""
        # VSCode HTML display
        if self.is_vscode and hasattr(self, '_use_html_display') and self._use_html_display:
            try:
                from IPython.display import HTML
                html = self._generate_html_display()
                if hasattr(self, '_html_display'):
                    self._html_display.update(HTML(html))
                return
            except:
                pass  # Fall through to tqdm

        if not self._tqdm_bars:
            return

        bar_idx = 0
        for unit in self.units:
            usage = self._current_usage.get(unit.name, [])

            for i, cpu_usage in enumerate(usage):
                if bar_idx < len(self._tqdm_bars):
                    bar = self._tqdm_bars[bar_idx]
                    # Update value and percentage
                    bar.n = cpu_usage
                    bar.set_postfix_str(f"{cpu_usage:.1f}%")
                    bar.refresh()
                    bar_idx += 1

        # Force display update for VSCode
        if self._has_ipython_display:
            try:
                from IPython.display import display
                # This helps VSCode update the display
                pass
            except:
                pass

    def _print_summary(self):
        """Print summary statistics."""
        if not self.show_summary:
            return

        # Don't print text summary if using HTML display in Jupyter/VSCode
        if self.is_jupyter and self.is_vscode and hasattr(self, '_use_html_display') and self._use_html_display:
            # HTML summary will be shown instead
            return

        print("\n" + "=" * 60)
        print("CPU Monitoring Summary")
        print("=" * 60)

        for unit in self.units:
            stats = self._stats[unit.name]
            summary = stats.get_summary()

            if not summary:
                continue

            # Display label based on grouping mode
            label = "Task" if self.group_by == "task" else "Node"
            print(f"\n{label}: {unit.name}")
            print(f"  Duration: {summary['duration']:.1f}s")
            print(f"  Overall: mean={summary['overall_mean']:.1f}%, "
                  f"max={summary['overall_max']:.1f}%, "
                  f"min={summary['overall_min']:.1f}%")
            print(f"  CPU-seconds: {summary['cpu_seconds']:.1f}")

            # Memory stats if available
            if 'memory_mean' in summary:
                print(f"  Memory: mean={summary['memory_mean']:.1f}%, "
                      f"max={summary['memory_max']:.1f}%, "
                      f"min={summary['memory_min']:.1f}%")

            # Per-core stats
            print(f"  Per-core averages:")
            mean_per_core = summary['mean_per_core']
            for i, mean_usage in enumerate(mean_per_core):
                max_usage = summary['max_per_core'][i]
                min_usage = summary['min_per_core'][i]
                print(f"    CPU {i:2d}: mean={mean_usage:5.1f}%, "
                      f"max={max_usage:5.1f}%, min={min_usage:5.1f}%")

        print("=" * 60)

    def start(self):
        """Start monitoring."""
        if self._monitoring:
            # logger.warning("Monitoring already started")
            return

        self._monitoring = True

        # Get initial CPU sample before starting display
        try:
            import socket
            current_hostname = _clean_hostname(socket.gethostname())
            our_unit = None

            # Find local unit
            for unit in self.units:
                if unit.is_local:
                    our_unit = unit
                    break

            # Fallback: check by hostname (for nodes) or process_id (for nodes)
            if not our_unit:
                for unit in self.units:
                    if self.group_by == "node":
                        if unit.name == current_hostname or getattr(unit, 'process_id', -1) == 0:
                            our_unit = unit
                            break
                    else:
                        # For tasks, is_local should have been set correctly
                        pass

            if our_unit:
                # Get initial CPU reading
                initial_usage = psutil.cpu_percent(interval=0.1, percpu=True)
                if our_unit.allocated_cpus:
                    try:
                        initial_usage = [initial_usage[i] for i in our_unit.allocated_cpus
                                        if i < len(initial_usage)]
                    except IndexError:
                        pass
                self._current_usage[our_unit.name] = initial_usage

                # Get initial memory reading in MB
                initial_memory_mb = psutil.virtual_memory().used / (1024 ** 2)
                self._current_memory[our_unit.name] = initial_memory_mb
        except Exception as e:
            # logger.debug(f"Could not get initial CPU sample: {e}")
            print(f"Could not get initial CPU sample: {e}")
            raise e

        # Start monitoring thread
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

        # Start display
        if self.is_jupyter:
            self._create_jupyter_display()

            # Start update loop in background
            def jupyter_update_loop():
                while self._monitoring:
                    self._update_jupyter_display()
                    time.sleep(self.update_interval)

            self._jupyter_update_thread = threading.Thread(target=jupyter_update_loop, daemon=True)
            self._jupyter_update_thread.start()
        else:
            # Terminal - use rich Live
            self._live = Live(self._create_terminal_display(),
                            console=self._console,
                            refresh_per_second=1.0/self.update_interval)
            self._live.start()

            # Update loop
            def terminal_update_loop():
                while self._monitoring:
                    if self._live:
                        self._live.update(self._create_terminal_display())
                    time.sleep(self.update_interval)

            self._terminal_update_thread = threading.Thread(target=terminal_update_loop, daemon=True)
            self._terminal_update_thread.start()

    def stop(self, had_error: bool = False):
        """Stop monitoring."""
        if not self._monitoring:
            return

        self._monitoring = False

        # Wait for monitor thread
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

        # Wait for jupyter update thread
        if self._jupyter_update_thread:
            self._jupyter_update_thread.join(timeout=2.0)

        # Stop display
        if self._live:
            self._live.stop()
            self._live = None

        # Close tqdm bars
        for bar in self._tqdm_bars:
            bar.close()
        self._tqdm_bars = []

        # Handle display based on persist and summary settings
        show_final_display = self.persist or self.summary

        if show_final_display:
            # Show summary (bars or table)
            if self.is_vscode and hasattr(self, '_use_html_display') and self._use_html_display:
                if hasattr(self, '_html_display'):
                    from IPython.display import HTML
                    summary_html = self._generate_html_display(summary_mode=True)
                    self._html_display.update(HTML(summary_html))
            # Print summary (will be skipped for VSCode HTML)
            self._print_summary()
        else:
            # Clear display if not persisting
            if self.is_vscode and hasattr(self, '_use_html_display') and self._use_html_display:
                if hasattr(self, '_html_display'):
                    try:
                        from IPython.display import clear_output
                        if not had_error:
                            clear_output(wait=False)
                    except:
                        pass

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop(exc_type is not None)
        return False


# ============================================================================
# Decorator
# ============================================================================

def monitor(func: Callable = None, **monitor_kwargs):
    """
    Decorator for CPU monitoring.

    Parameters
    ----------
    func : callable
        Function to monitor
    **monitor_kwargs
        Arguments passed to CPUMonitor

    Examples
    --------
    >>> @monitor
    >>> def my_function():
    ...     # computation
    ...     pass

    >>> @monitor(width=100, update_interval=1.0)
    >>> def my_function():
    ...     # computation
    ...     pass
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            with CPUMonitor(**monitor_kwargs):
                return f(*args, **kwargs)
        return wrapper

    if func is None:
        # Called with arguments: @monitor(...)
        return decorator
    else:
        # Called without arguments: @monitor
        return decorator(func)


# ============================================================================
# IPython Magic
# ============================================================================


from IPython.core.magic import Magics, magics_class, cell_magic
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring

@magics_class
class CPUMonitorMagics(Magics):
    """IPython magics for CPU monitoring."""

    @cell_magic
    @magic_arguments()
    @argument('--width', '-w', type=int, default=None,
                help='Display width in characters')
    @argument('--interval', '-i', type=float, default=0.5,
                help='Update interval in seconds')
    @argument('--persist', '-p', action='store_true',
                help='Keep display visible after completion')
    @argument('--color', '-c', action='store_true',
                help='Use color coding (green/yellow/red). Default is gray only.')
    @argument('--summary', '-s', action='store_true',
                help='Show summary table with mean CPU usage per core (mean/max memory shown next to node name)')
    @argument('--fold', '-f', type=int, default=16,
                help='Maximum number of CPU bars per row before wrapping (default: 16)')
    @argument('--group-by', '-g', type=str, default='node', choices=['node', 'task'],
                help='Group CPU bars by "node" (default) or "task"')
    @argument('--per-node-layout', '-n', action='store_true',
                help='Use per-node layout for CPU bars')
    def monitor(self, line, cell):
        """
        Monitor CPU usage during cell execution.

        Usage:
            %%monitor
            # your code here

            %%monitor --width 100 --interval 1.0
            # your code here

            %%monitor --persist
            # display remains after completion

            %%monitor --color
            # use color coding (green/yellow/red)

            %%monitor --summary
            # show table with CPU and memory statistics

            %%monitor --group-by task
            # group by SLURM task instead of node
        """
        args = parse_argstring(self.monitor, line)

        monitor = CPUMonitor(width=args.width, update_interval=args.interval,
                        persist=args.persist, color=args.color, summary=args.summary,
                        fold=args.fold, group_by=args.group_by)

        monitor.start()
        try:
            # Execute cell and capture result
            result = self.shell.run_cell(cell)
        finally:
            # Check if cell execution had an error
            had_error = hasattr(result, 'error_in_exec') and result.error_in_exec is not None

            # If there was an error, don't clear the display (preserve error output)
            if had_error:
                # Stop monitoring but don't clear display
                monitor._monitoring = False
                if monitor._monitor_thread:
                    monitor._monitor_thread.join(timeout=2.0)
                if monitor._jupyter_update_thread:
                    monitor._jupyter_update_thread.join(timeout=2.0)
                if monitor._live:
                    monitor._live.stop()
                    monitor._live = None
                for bar in monitor._tqdm_bars:
                    bar.close()
                monitor._tqdm_bars = []
                # Don't update or clear the HTML display - let error show
            else:
                # Normal stop with display update/clear
                monitor.stop()

# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    'CPUMonitor',
    'monitor_cpu',
    'CPUMonitorMagics',
    'NodeInfo',
    'TaskInfo',
    'detect_compute_nodes',
    'detect_compute_tasks',
    'get_cached_nodes',
    'get_cached_tasks',
    'is_jupyter',
    'is_vscode',
]
