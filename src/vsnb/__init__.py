
from .plot_theme import set_vscode_theme
from .plot_theme import vscode_theme
from .progress import pqdm, prange
# CPU monitoring
from .cpu_monitor import (
    CPUMonitor,
    monitor,
    CPUMonitorMagics,
    detect_compute_nodes,
    get_cached_nodes,
)


try:
    from IPython import get_ipython
    ipython = get_ipython()
    if ipython is not None and CPUMonitorMagics is not None:
        ipython.register_magics(CPUMonitorMagics)
except (ImportError, NameError):
    # Not in IPython environment or magic not available
    pass