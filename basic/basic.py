"""
Basic module: Original logic for reading and rendering text inside blocks.
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from utils import get_random_font_path, extract_bits_with_positions, group_bits_by_lines, recreate_shape_from_bits, process_image_to_shape

__all__ = [
    'calculate_margins',
    'render_text_image',
    'process_basic',
    'extract_bits_with_positions',
    'group_bits_by_lines',
    'recreate_shape_from_bits'
]

def calculate_margins(blocks, canvas_width, canvas_height):
    left_margin = min(b['x0'] for b in blocks)
    top_margin = min(b['y0'] for b in blocks)
    right_margin = canvas_width - max(b['x1'] for b in blocks)
    bottom_margin = canvas_height - max(b['y1'] for b in blocks)
    return {
        'left': left_margin,
        'top': top_margin,
        'right': right_margin,
        'bottom': bottom_margin
    }

def render_text_image(lines, bits, decoded_text, original_image_size, output_file='output/translated_preview.png', config=None):
    if not bits:
        print("[DEBUG] No bits/blocks to render")
        return
    print(f"[DEBUG] Rendering {len(bits)} bits/blocks with text: '{decoded_text[:50]}...'")
    min_x = min(b['x'] for b in bits)
    min_y = min(b['y'] for b in bits)
    max_x = max(b['x'] + b['w'] for b in bits)
    max_y = max(b['y'] + b['h'] for b in bits)
    content_width = max_x - min_x
    content_height = max_y - min_y
    print(f"[DEBUG] Content bounds: ({min_x},{min_y}) to ({max_x},{max_y}), size: {content_width}x{content_height}")
    orig_width, orig_height = original_image_size
    blocks = [{'x0': b['x'], 'y0': b['y'], 'x1': b['x'] + b['w'], 'y1': b['y'] + b['h']} for b in bits]
    left = top = right = bottom = 0
    if config:
        from basic.basic import calculate_margins
        margins = calculate_margins(blocks, orig_width, orig_height)
        left, top, right, bottom = margins['left'], margins['top'], margins['right'], margins['bottom']
    canvas_width = int(content_width + left + right)
    canvas_height = int(content_height + top + bottom)
    print(f"[DEBUG] Canvas size: {canvas_width}x{canvas_height}")
    canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    idx = 0
    full_text = decoded_text * 1000
    chars_drawn = 0
    for line in lines:
        if not line:
            continue
        line_text = ''
        for _ in line:
            if idx >= len(full_text):
                break
            line_text += full_text[idx]
            idx += 1
        if not line_text:
            continue
        x_positions = [bit['x'] for bit in line]
        min_line_x = min(x_positions)
        max_line_x = max(x_positions)
        line_width = max_line_x - min_line_x if len(x_positions) > 1 else 20
        font_size = 12
        max_font_size = 60
        # Font selection logic
        if config and config.get('font_path'):
            font_path = config['font_path']
        else:
            font_path = get_random_font_path()
        if font_path:
            try:
                base_font = ImageFont.truetype(font_path, 12)
            except:
                base_font = ImageFont.load_default()
        else:
            base_font = ImageFont.load_default()
        for size in range(10, max_font_size):
            if font_path:
                try:
                    font = ImageFont.truetype(font_path, size)
                except:
                    font = base_font
            else:
                font = base_font
            bbox = font.getbbox(line_text)
            text_width = bbox[2] - bbox[0]
            if text_width > line_width * 0.95:
                font_size = size - 1
                break
            font_size = size
        if font_path:
            try:
                font = ImageFont.truetype(font_path, font_size)
            except:
                font = base_font
        else:
            font = base_font
        for i, bit in enumerate(line):
            x = int(bit['x'] - min_x + left)
            y = int(bit['y'] - min_y + top)
            bgr = bit.get('color', (0, 0, 0))
            rgb = (bgr[2], bgr[1], bgr[0])
            color = rgb
            char = line_text[i] if i < len(line_text) else ' '
            if 0 <= x < canvas_width and 0 <= y < canvas_height:
                draw.text((x, y), char, fill=color, font=font)
                chars_drawn += 1
    print(f"[DEBUG] Drew {chars_drawn} characters")
    canvas.save(output_file)
    print(f"[DEBUG] Saved to {output_file}")

def render_blocks_image(blocks, decoded_text, original_image_size, output_file='output/translated_blocks.png', config=None):
    """Render text in colored blocks (for Basic mode)"""
    if not blocks:
        return
    
    # Calculate content bounds
    min_x = min(b['x'] for b in blocks)
    min_y = min(b['y'] for b in blocks)
    max_x = max(b['x'] + b['w'] for b in blocks)
    max_y = max(b['y'] + b['h'] for b in blocks)
    content_width = max_x - min_x
    content_height = max_y - min_y
    
    orig_width, orig_height = original_image_size
    
    # Apply margins
    blocks_for_margins = [{'x0': b['x'], 'y0': b['y'], 'x1': b['x'] + b['w'], 'y1': b['y'] + b['h']} for b in blocks]
    margins = calculate_margins(blocks_for_margins, orig_width, orig_height)
    left, top, right, bottom = margins['left'], margins['top'], margins['right'], margins['bottom']
    
    canvas_width = int(content_width + left + right)
    canvas_height = int(content_height + top + bottom)
    
    canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    # Font selection logic
    if config and config.get('font_path'):
        font_path = config['font_path']
    else:
        font_path = get_random_font_path()
    
    if font_path:
        try:
            base_font = ImageFont.truetype(font_path, 12)
        except:
            base_font = ImageFont.load_default()
    else:
        base_font = ImageFont.load_default()
    
    full_text = decoded_text * 1000
    text_idx = 0
    
    for i, block in enumerate(blocks):
        x = int(block['x'] - min_x + left)
        y = int(block['y'] - min_y + top)
        w = int(block['w'])
        h = int(block['h'])
        color = tuple(int(c) for c in block['color']) if isinstance(block['color'], (tuple, list)) else (0, 0, 0)
        
        # Get character for this block
        if text_idx >= len(full_text):
            text_idx = 0
        char = full_text[text_idx]
        text_idx += 1
        
        # Ensure we have a printable character
        if not char.isprintable():
            char = 'A'  # Fallback to a safe character
        
        # Find appropriate font size
        font_size = 10
        max_font_size = min(w, h)
        for size in range(10, max_font_size):
            if font_path:
                try:
                    font = ImageFont.truetype(font_path, size)
                except:
                    font = base_font
            else:
                font = base_font
            bbox = font.getbbox(char)
            char_w = bbox[2] - bbox[0]
            char_h = bbox[3] - bbox[1]
            if char_w <= w * 0.8 and char_h <= h * 0.8:
                font_size = size
            else:
                break
        
        if font_path:
            try:
                font = ImageFont.truetype(font_path, font_size)
            except:
                font = base_font
        else:
            font = base_font
        
        # Center text in block
        bbox = font.getbbox(char)
        char_w = bbox[2] - bbox[0]
        char_h = bbox[3] - bbox[1]
        text_x = int(x + (w - char_w) // 2)
        text_y = int(y + (h - char_h) // 2)
        
        # Draw character with the block's color
        rgb = (color[2], color[1], color[0]) if len(color) == 3 else (0, 0, 0)
        text_color = rgb  # Use the block's color for text
        if 0 <= text_x < canvas_width and 0 <= text_y < canvas_height:
            draw.text((text_x, text_y), char, fill=text_color, font=font)
    
    canvas.save(output_file)

def process_basic(image_path, config, decoded_text, features=None):
    # Use provided features or extract if not provided
    if features is None:
        decoded_text, shape_output, features = process_image_to_shape(image_path, config=config)
    bits = features.get('bits', [])
    blocks = features.get('blocks', [])
    lines = features.get('lines', [])
    original_image_size = (features['color_img'].shape[1], features['color_img'].shape[0]) if features.get('color_img') is not None else (0, 0)
    # Use blocks for basic mode (colored regions) instead of bits (binary positions)
    if blocks:
        render_blocks_image(blocks, decoded_text, original_image_size, output_file="output/final_output_basic.jpeg", config=config)
    else:
        render_text_image(lines, bits, decoded_text, original_image_size, output_file="output/final_output_basic.jpeg", config=config) 