"""
Configuration file for paintBinary project.
Edit these settings to control behavior of Posterlike and Basic modules.
"""
CONFIG = {
    # Font settings
    'font_path': None,  # If None, randomize from system fonts. If set, use only this font.
    'min_font_pt': 11,
    'max_font_pt': 250,
    # Macroblock merging
    'macroblock_proximity_thresh': 120,
    'macroblock_color_thresh': 20,
    'min_area': 50,
    # Posterlike-specific
    'large_block_threshold': 2000,
    # Other settings can be added here
} 