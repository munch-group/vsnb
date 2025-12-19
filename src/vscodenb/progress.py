
import time
from threading import local

# Thread-local storage for nested progress bar tracking
_thread_local = local()


def _get_nesting_stack():
    """Get or create the nesting stack for current thread."""
    if not hasattr(_thread_local, 'nesting_stack'):
        _thread_local.nesting_stack = []
    return _thread_local.nesting_stack


def _get_display_handles():
    """Get or create the display handles dict for current thread."""
    if not hasattr(_thread_local, 'display_handles'):
        _thread_local.display_handles = {}
    return _thread_local.display_handles


def _get_persist_flag():
    """Get or create the persist flag for current thread."""
    if not hasattr(_thread_local, 'should_persist'):
        _thread_local.should_persist = False
    return _thread_local.should_persist


def _set_persist_flag(value):
    """Set the persist flag for current thread."""
    _thread_local.should_persist = value


def _is_notebook():
    """Detect if running in Jupyter/VS Code notebook environment or nbconvert execution."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False

        shell_type = str(type(shell))

        # Terminal IPython should use tqdm fallback
        if 'TerminalInteractiveShell' in shell_type:
            return False

        # ZMQInteractiveShell = Jupyter notebook/lab, VSCode notebooks
        # Other non-terminal shells (like nbconvert's kernel) also support HTML display
        return True

    except (ImportError, NameError):
        return False


class HTMLProgressBar:
    """
    HTML-based progress bar for notebooks with tqdm fallback for terminal.
    Renders as thin, sleek bars matching cpu_monitor.py style in VS Code/Jupyter.
    """

    def __init__(self, iterable=None, total=None, desc='', color=None, persist=False, **kwargs):
        self.iterable = iterable
        self.total = total if total is not None else (len(iterable) if iterable is not None and hasattr(iterable, '__len__') else None)
        self.desc = desc
        self.color = color
        self.persist = persist
        self.n = 0
        self.start_time = time.time()
        self._display_handle = None
        self._use_html = _is_notebook()

        # Track nesting level
        self._nesting_level = None
        self._owns_nesting_level = False

        # For terminal fallback
        if not self._use_html:
            from tqdm import tqdm
            # Pass leave=persist to tqdm for consistent behavior
            self._tqdm = tqdm(iterable=iterable, total=total, desc=desc, leave=persist, **kwargs)
        else:
            self._tqdm = None
            self._initialize_display()

    def _initialize_display(self):
        """Initialize HTML display in notebook, reusing widgets for nested loops."""
        if self._use_html:
            try:
                from IPython.display import display, HTML

                # Track if any progress bar in this cell wants to persist
                if self.persist:
                    _set_persist_flag(True)

                # Determine nesting level
                nesting_stack = _get_nesting_stack()
                self._nesting_level = len(nesting_stack)

                # Check if we have an existing display handle for this level
                display_handles = _get_display_handles()

                if self._nesting_level in display_handles:
                    # Reuse existing display handle
                    self._display_handle = display_handles[self._nesting_level]
                    # Reset state for reused widget
                    self.n = 0
                    self.start_time = time.time()
                else:
                    # Create new display handle
                    html = self._generate_html()
                    self._display_handle = display(HTML(html), display_id=True)
                    display_handles[self._nesting_level] = self._display_handle

                # Update the display immediately
                self._display_handle.update(HTML(self._generate_html()))

            except ImportError:
                # Fallback if IPython not available
                self._use_html = False

    def _generate_html(self):
        """Generate HTML for progress bar matching cpu_monitor.py style."""
        if self.total and self.total > 0:
            percentage = min(100, max(0, (self.n / self.total) * 100))
        else:
            percentage = 0

        # Calculate elapsed time and rate
        elapsed = time.time() - self.start_time
        rate = self.n / elapsed if elapsed > 0 else 0

        # Estimate remaining time with adaptive units
        if self.total and rate > 0 and self.n > 0 and self.n < self.total:
            remaining = (self.total - self.n) / rate
            if remaining >= 3600:
                remaining_str = f"{remaining / 3600:.1f} h"
            elif remaining >= 60:
                remaining_str = f"{remaining / 60:.1f} m"
            else:
                remaining_str = f"{remaining:.0f} s"
        elif self.n >= self.total:
            remaining_str = "0 s"
        else:
            remaining_str = "?"

        # Progress info
        if self.total:
            progress_text = f"{self.n}/{self.total}"
        else:
            progress_text = f"{self.n}"

        # Determine color based on percentage (matching cpu_monitor.py lines 1312-1322)
        if self.color is None:
            # Default: monochrome gray (matching cpu_monitor.py default)
            bar_color = '#666666'  # gray
        elif self.color == 'auto':
            # Auto color mode: green/yellow/red based on percentage
            if percentage < 50:
                bar_color = '#4CAF50'  # green
            elif percentage < 80:
                bar_color = '#FFC107'  # yellow
            else:
                bar_color = '#F44336'  # red
        else:
            # Use specified color
            bar_color = self.color

        # # Build HTML (matching cpu_monitor.py style at lines 1203, 1327-1329)
        # html = f'''
        # <div style="font-family: monospace; font-size: 10px; padding: 0px 10px; display: flex; align-items: center; gap: 10px;">
        #     <div style="width: 2ch; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
        #         {self.desc} {percentage:.0f}% | {progress_text} | {remaining_str}
        #     </div>
        #     <div style="flex: 1; height: 8px; background: rgba(128, 128, 128, 0.2); border-radius: 2px; overflow: hidden;">
        #         <div style="width: {percentage}%; height: 100%; background: {bar_color}; transition: width 0.3s;"></div>
        #     </div>
        # </div>
        # '''
        # Build HTML (matching cpu_monitor.py style at lines 1203, 1327-1329)
        html = f'''
        <div style="font-family: monospace; font-size: 10px; padding: 0px 10px, 5px 10px; display: flex; align-items: center; gap: 10px;">
            <div style="width: 15ch; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                {self.desc} {percentage:.0f}% | {remaining_str}
            </div>
            <div style="flex: 1; height: 8px; background: rgba(128, 128, 128, 0.2); border-radius: 2px; overflow: hidden;">
                <div style="width: {percentage}%; height: 100%; background: {bar_color}; transition: width 0.3s;"></div>
            </div>
        </div>
        '''
        return html

    def update(self, n=1):
        """Update progress by n steps."""
        if self._tqdm is not None:
            self._tqdm.update(n)
        else:
            self.n += n
            if self._display_handle is not None:
                from IPython.display import HTML
                self._display_handle.update(HTML(self._generate_html()))

    def _clear_display(self):
        """Remove the output cell entirely."""
        if self._use_html:
            from IPython.display import clear_output
            clear_output(wait=False)

    def __iter__(self):
        """Iterate over the iterable, updating progress."""
        if self._tqdm is not None:
            return iter(self._tqdm)

        if self.iterable is None:
            raise ValueError("iterable must be provided for iteration")

        # Push this progress bar onto the nesting stack
        nesting_stack = _get_nesting_stack()
        nesting_stack.append(self)
        self._owns_nesting_level = True

        try:
            for item in self.iterable:
                yield item
                self.update(1)
        finally:
            # Pop from nesting stack when iteration completes
            if self._owns_nesting_level and nesting_stack and nesting_stack[-1] is self:
                nesting_stack.pop()
                self._owns_nesting_level = False
                # When all loops complete, clear output if no progress bar had persist=True
                if len(nesting_stack) == 0:
                    if not _get_persist_flag():
                        self._clear_display()
                    _get_display_handles().clear()
                    _set_persist_flag(False)

    def __enter__(self):
        """Context manager entry."""
        if self._tqdm is not None:
            return self._tqdm.__enter__()

        # Push onto nesting stack if not already there
        nesting_stack = _get_nesting_stack()
        if not self._owns_nesting_level:
            nesting_stack.append(self)
            self._owns_nesting_level = True

        return self

    def __exit__(self, *args):
        """Context manager exit."""
        if self._tqdm is not None:
            return self._tqdm.__exit__(*args)

        # Finalize HTML display to show completed state
        if self._display_handle is not None:
            from IPython.display import HTML
            self.n = self.total if self.total else self.n
            self._display_handle.update(HTML(self._generate_html()))

        # Pop from nesting stack
        nesting_stack = _get_nesting_stack()
        if self._owns_nesting_level and nesting_stack and nesting_stack[-1] is self:
            nesting_stack.pop()
            self._owns_nesting_level = False
            # When all loops complete, clear output if no progress bar had persist=True
            if len(nesting_stack) == 0:
                if not _get_persist_flag():
                    self._clear_display()
                _get_display_handles().clear()
                _set_persist_flag(False)

    def close(self):
        """Close the progress bar."""
        if self._tqdm is not None:
            self._tqdm.close()
        else:
            # Pop from nesting stack if we own a level
            nesting_stack = _get_nesting_stack()
            if self._owns_nesting_level and nesting_stack and nesting_stack[-1] is self:
                nesting_stack.pop()
                self._owns_nesting_level = False
                # When all loops complete, clear output if no progress bar had persist=True
                if len(nesting_stack) == 0:
                    if not _get_persist_flag():
                        self._clear_display()
                    _get_display_handles().clear()
                    _set_persist_flag(False)


def pqdm(iterable=None, total=None, desc='', persist=False, **kwargs):
    """
    HTML progress bar for notebooks, tqdm for terminal.
    Renders as thin bar matching cpu_monitor.py style.

    Args:
        iterable: The iterable to iterate over.
        total: Total number of items (optional if iterable has __len__).
        desc: Description text shown next to the progress bar.
        persist: If False (default), the progress bar disappears after completion.
                 If True, the progress bar remains visible in its completed state.
        **kwargs: Additional arguments passed to tqdm (terminal fallback).

    Usage:
        for item in pqdm(iterable, desc="Processing"):
            process(item)
    """
    return HTMLProgressBar(iterable=iterable, total=total, desc=desc, persist=persist, **kwargs)


def prange(n, desc='', persist=False, **kwargs):
    """
    HTML progress bar for range() in notebooks, tqdm for terminal.
    Renders as thin bar matching cpu_monitor.py style.

    Args:
        n: The number to iterate up to (like range(n)).
        desc: Description text shown next to the progress bar.
        persist: If False (default), the progress bar disappears after completion.
                 If True, the progress bar remains visible in its completed state.
        **kwargs: Additional arguments passed to tqdm (terminal fallback).

    Usage:
        for i in prange(100, desc="Processing"):
            process(i)
    """
    return HTMLProgressBar(iterable=range(n), total=n, desc=desc, persist=persist, **kwargs)
