import numpy as np
import json5 as json
import os
import platform
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib_inline.backend_inline import set_matplotlib_formats
import seaborn as sns
# import colorsys
import os
import sys
from pathlib import Path
from typing import Any, TypeVar, List, Tuple, Dict, Union
from collections.abc import Sequence, MutableSequence, Callable
from itertools import cycle
from IPython.display import display, HTML

# def random_color():
#     return '#'+''.join(random.sample('0123456789ABCDEF', 6))

def _get_color(n, lightness=0.4):
    color_cycle = cycle([matplotlib.colors.to_hex(c) for c in sns.husl_palette(n, l=lightness)])
    for color in color_cycle:
        yield color
        
def rgb_to_hsl(r, g, b):
    r = float(r)
    g = float(g)
    b = float(b)
    high = max(r, g, b)
    low = min(r, g, b)
    h, s, v = ((high + low) / 2,)*3

    if high == low:
        h = 0.0
        s = 0.0
    else:
        d = high - low
        s = d / (2 - high - low) if l > 0.5 else d / (high + low)
        h = {
            r: (g - b) / d + (6 if g < b else 0),
            g: (b - r) / d + 2,
            b: (r - g) / d + 4,
        }[high]
        h /= 6

    return h, s, v


def hsl_to_rgb(h, s, l):
    def hue_to_rgb(p, q, t):
        t += 1 if t < 0 else 0
        t -= 1 if t > 1 else 0
        if t < 1/6: return p + (q - p) * 6 * t
        if t < 1/2: return q
        if t < 2/3: p + (q - p) * (2/3 - t) * 6
        return p

    if s == 0:
        r, g, b = l, l, l
    else:
        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        r = hue_to_rgb(p, q, h + 1/3)
        g = hue_to_rgb(p, q, h)
        b = hue_to_rgb(p, q, h - 1/3)

    return r, g, b

class suppress_plotting_output:
    def __init__(self):
        pass

    def __enter__(self):
        plt.ioff()

    def __exit__(self, exc_type, exc_value, traceback):
        plt.ion()

    
def get_vscode_settings_path() -> Path:
    """Get the path to VS Code's user settings.json based on the OS."""
    system = platform.system()
    
    if system == "Darwin":  # macOS
        return Path.home() / "Library" / "Application Support" / "Code" / "User" / "settings.json"
    elif system == "Windows":
        return Path(os.environ.get("APPDATA", "")) / "Code" / "User" / "settings.json"
    else:  # Linux
        return Path.home() / ".config" / "Code" / "User" / "settings.json"


def get_vscode_theme_from_config() -> str | None:
    """Parse VS Code settings and return the workbench.colorTheme value."""
    settings_path = get_vscode_settings_path()
    
    if settings_path.exists():

        with open(settings_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Remove single-line comments (// ...) and multi-line comments (/* ... */)
        # This handles JSONC (JSON with Comments)
        import re
        content = re.sub(r'//.*?$', '', content, flags=re.MULTILINE)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        settings = json.loads(content)
        return settings.get("workbench.colorTheme", "Default Dark Modern")
    else:
        # print(f"No settings file not found at: {settings_path}")
        return None


def is_vscode_dark_theme(mode=None) -> bool:
    """Determine if a given VS Code theme name is dark or light."""

    is_dark = None

    env_theme = os.environ.get('NOTEBOOK_THEME', None)
    if env_theme is not None:
        print("Overriding theme from NOTEBOOK_THEME environment variable.", sys.stderr)
        return 'dark' in env_theme.lower()

    if mode is not None:
        is_dark = 'dark' in mode.lower()
    else:
        theme = get_vscode_theme_from_config()
        if theme is not None:
            is_dark = 'dark' in theme.lower()

    if is_dark is None:
        with suppress_plotting_output():
            # make a plot to check background color
            fig, ax = plt.subplots()
            bg_color = ax.get_facecolor()
            plt.close(fig)
            luminance = matplotlib.colors.rgb_to_hsv(matplotlib.colors.to_rgb(bg_color))[2]
            is_dark = luminance < 0.5

    return is_dark


def lighten_colors(colors, factor=0.0, n_colors=None, as_cmap=None, target_lightness=None):
    """
    Lighten a colormap or palette using HSL color space.
    
    Parameters
    ----------
    colors : str, Colormap, or list
        Matplotlib colormap, seaborn palette name, or list of colors
    factor : float
        Blend factor toward white (0 = original, 1 = white).
        Ignored if target_lightness is set.
    n_colors : int, optional
        Number of colors for discrete palettes
    as_cmap : bool, optional
        Force output type. If None, auto-detects based on input.
    target_lightness : float, optional
        If set (0-1), all colors will be adjusted to this lightness value,
        preserving hue and saturation.
        
    Returns
    -------
    list or LinearSegmentedColormap
    """
    is_cmap_input = False
    
    # Handle different input types
    if isinstance(colors, matplotlib.colors.Colormap):
        is_cmap_input = True
        color_list = colors(np.linspace(0, 1, 256))
        name = colors.name
    elif isinstance(colors, str):
        # Check if n_colors is specified or name suggests a discrete palette
        discrete_names = {'tab10', 'tab20', 'tab20b', 'tab20c', 'Set1', 'Set2', 'Set3',
                          'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
                          'deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind'}
        
        if n_colors is not None or colors in discrete_names:
            # Treat as discrete palette - try seaborn first
            try:
                import seaborn as sns
                color_list = sns.color_palette(colors, n_colors=n_colors)
                name = colors
                is_cmap_input = False
            except:
                # Fall back to matplotlib
                cmap = plt.get_cmap(colors)
                n = n_colors or 10
                color_list = [cmap(i / (n - 1)) for i in range(n)]
                name = colors
                is_cmap_input = False
        else:
            # Try as continuous matplotlib colormap
            try:
                cmap = plt.get_cmap(colors)
                is_cmap_input = True
                color_list = cmap(np.linspace(0, 1, 256))
                name = colors
            except ValueError:
                # Fall back to seaborn palette
                import seaborn as sns
                color_list = sns.color_palette(colors, n_colors=n_colors)
                name = colors
    else:
        # Assume it's a list of colors
        color_list = list(colors)
        name = "custom"
    
    # Process each color in HSL space
    lightened = []
    for c in color_list:
        rgba = matplotlib.colors.to_rgba(c)
        r, g, b, a = rgba
        
        # h, l, s = colorsys.rgb_to_hls(r, g, b)
        h, s, l = rgb_to_hsl(r, g, b)
        
        if target_lightness is not None:
            l_new = target_lightness
        else:
            l_new = l + (1 - l) * factor
        
        # r_new, g_new, b_new = colorsys.hls_to_rgb(h, l_new, s)
        r_new, g_new, b_new = hsl_to_rgb(h, s, l_new)
        lightened.append((r_new, g_new, b_new, a))
    
    # Determine output type
    if as_cmap is None:
        as_cmap = is_cmap_input
    
    if as_cmap:
        return matplotlib.colors.LinearSegmentedColormap.from_list(f"{name}_light", lightened)
    
    return lightened


# Add a style kwarg to vscode_theme: grid, ticks
# and a frame kwarg that can be True/False
# theme() standard plot
# theme(style='grid') standard plot
# theme(style='grid', frame=True) grid with frame
# theme(style='ticks') despined ticks style 
# theme(style='ticks', frame=True)

def set_vscode_theme(mode=None, style='grid', frame=False, cmap=None, figsize=(5, 3.7), font_scale=1.0):
    """
    Set the default theme for the graph plotter.
    The theme can be either 'dark' or 'light'. The default theme is autodetected.
    NOTEBOOK_THEME environment variable can be used to override the theme.

    Parameters
    ----------
    mode : str, optional
        'dark' or 'light' to force a specific theme. If None, autodetected.
    style : str, optional
        'grid' for grid style, 'ticks' for ticks style.
    frame : bool, optional
        Whether to show frame around plots.
    cmap : Colormap, optional
        Matplotlib colormap to use for plots.
    figsize : tuple, optional
        Default figure size.
    font_scale : float, optional
        Scale factor for fonts.
    """

    with suppress_plotting_output():

        is_dark = is_vscode_dark_theme(mode=mode)

        set_matplotlib_formats('retina', 'png')

        sns.set_context('paper', font_scale=font_scale)

        # if cmap is None:
        #     cmap = plt.get_cmap()

        # # dark_cmap = lighten_colors(cmap, target_lightness=0.4, as_cmap=True)
        # # light_cmap = lighten_colors(cmap, target_lightness=0.6, as_cmap=True)
        # dark_cmap = lighten_colors(cmap, factor=0, as_cmap=True)
        # light_cmap = lighten_colors(cmap, factor=0, as_cmap=True)

        if is_dark:
            plt.style.use('dark_background')
        else:
            plt.style.use('default')

        if is_dark:
            plt.rcParams.update({
                'figure.facecolor': '#1F1F1F', 
                'axes.facecolor': '#1F1F1F',
                'grid.linewidth': 0.4,
                'grid.alpha': 0.3,
                'grid.color': '#ffffff',            
                })
            # plt.set_cmap(cmap if cmap else dark_cmap)        
        else:
            plt.rcParams.update({
                'figure.facecolor': 'white', 
                'axes.facecolor': 'white',
                'grid.color': '#000000',
                'grid.linewidth': 0.4,
                'grid.alpha': 0.3,            
                })
            # plt.set_cmap(cmap if cmap else light_cmap)

        plt.rcParams.update({
            'axes.grid': style == 'grid',
            'axes.grid.axis': 'both',
            'axes.grid.which': 'major',
            'axes.titlelocation': 'right',
            'axes.titlesize': 'large',
            'axes.titleweight': 'normal',
            'axes.labelsize': 'medium',
            'axes.labelweight': 'normal',
            'axes.formatter.use_mathtext': True,
            'axes.spines.left': frame or style != 'grid',
            'axes.spines.bottom': frame or style != 'grid',
            'axes.spines.top': frame,
            'axes.spines.right':  frame,
            'xtick.bottom': style != 'grid',
            'ytick.left': style != 'grid',
            'legend.frameon': False,
            'figure.figsize': figsize,            
        })

        # Apply CSS to make ipywidget backgrounds transparent and match VS Code theme
        display(HTML("""
        <style>
        .cell-output-ipywidget-background {
            background-color: transparent !important;
        }
        :root {
            --jp-widgets-color: var(--vscode-editor-foreground);
            --jp-widgets-font-size: var(--vscode-editor-font-size);
        }  
        </style>                          
        """))


def black_white(ax):
    """Returns black for light backgrounds, white for dark backgrounds."""
    if ax is None:
        ax = plt.gca()
    bg_color = ax.get_facecolor()
    # Convert to grayscale to determine brightness
    luminance = matplotlib.colors.rgb_to_hsv(matplotlib.colors.to_rgb(bg_color))[2]
    return 'black' if luminance > 0.5 else '#FDFDFD'

class vscode_theme:
    def __init__(self, mode=None, style='grid', frame=False, **theme_kwargs):
        self.mode = mode
        self.style = style
        self.frame = frame
        self.theme_kwargs = theme_kwargs
        self.orig_rcParams = matplotlib.rcParams.copy()
        self.orig_cmap = matplotlib.pyplot.get_cmap()
        
    def __enter__(self):
        set_vscode_theme(mode=self.mode, style=self.style, frame=self.frame, **self.theme_kwargs)

    def __exit__(self, exc_type, exc_value, traceback):
        matplotlib.rcParams.update(self.orig_rcParams)
        matplotlib.pyplot.set_cmap(self.orig_cmap)

