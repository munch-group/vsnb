import shutil
import sys

from .plot_theme import set_vscode_theme, vscode_theme, lighten_colors, truncate_colormap
from .progress import pqdm, prange  
# CPU monitoring
from .cpu_monitor import (
    CPUMonitor,
    monitor,
    CPUMonitorMagics,
    detect_compute_nodes,
    get_cached_nodes,
)
from .slurm_magic import SlurmMagic
from .utils import truncate_colormap, Left

try:
    from IPython import get_ipython
    ipython = get_ipython()
    if ipython is not None and CPUMonitorMagics is not None:
        ipython.register_magics(CPUMonitorMagics)
    if ipython is not None and SlurmMagic is not None:
        ipython.register_magics(SlurmMagic)
except (ImportError, NameError):
    print("IPython not available, skipping magic registration.", file=sys.stderr)
    # Not in IPython environment or magic not available
    pass


from IPython.display import display, HTML

# # Apply CSS to make ipywidget backgrounds transparent and match VS Code theme
# display(HTML("""
# <style>
# .cell-output-ipywidget-background {
#     background-color: transparent !important;
# }
# :root {
#     --jp-widgets-color: var(--vscode-editor-foreground);
#     --jp-widgets-font-size: var(--vscode-editor-font-size);
# }  
# </style>                          
# """))

# make tqdm progress bar less intrusive:
html_style = '''
<style>
.widget-label,
.widget-html,
.widget-button,
.widget-dropdown,
.widget-text,
.widget-textarea {
    color: #CCCCCC !important;  /* Set to desired color */
    font-size: 10px !important;      /* Font size */

}
div.widget-html-content > progress { /* Outer container */
    height: 10px !important;

}
div.jp-OutputArea-output td.output_html { /* tqdm HTML output area */
    height: 5px !important;
}
div.progress { /* Jupyter Notebook (classic) */
    height: 5px !important;
    min-height: 5px !important;
    margin-top: 12px !important;    
}
div.progress-bar { /* Inner bar */
    height: 5px !important;
    min-height: 5px !important;
    line-height: 5px !important;
    color: 'white' !important;
    padding-bottom: 0px !important;    
}
</style>
'''
display(HTML(html_style))