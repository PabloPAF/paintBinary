"""
Basic module: Original logic for reading and rendering text inside blocks.
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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

def extract_bits_with_positions(thresh_img, color_img):
    import pytesseract
    config = r'--oem 3 --psm 11 -c user_defined_dpi=300 tessedit_char_whitelist=01'
    h_img, w_img = thresh_img.shape
    boxes = pytesseract.image_to_boxes(thresh_img, config=config)
    bits = []
    for b in boxes.splitlines():
        try:
            ch, x1, y1, x2, y2, _ = b.split()
            if ch not in ['0', '1']:
                continue
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            y1 = h_img - y1
            y2 = h_img - y2
            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            cx = min(max(x + w // 2, 0), w_img - 1)
            cy = min(max(y + h // 2, 0), h_img - 1)
            color = tuple(color_img[cy, cx].tolist())
            bits.append({
                'bit': ch,
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'color': color
            })
        except ValueError:
            continue
    return bits

def group_bits_by_lines(bits, band_height=20):
    if not bits:
        return []
    bands = {}
    for bit in bits:
        band_key = (bit['y'] // band_height) * band_height
        bands.setdefault(band_key, []).append(bit)
    sorted_band_keys = sorted(bands.keys())
    lines = []
    for key in sorted_band_keys:
        line = sorted(bands[key], key=lambda b: b['x'])
        lines.append(line)
    return lines

def render_text_image(lines, bits, decoded_text, original_image_size, output_file='output/translated_preview.png'):
    if not bits:
        return
    min_x = min(b['x'] for b in bits)
    min_y = min(b['y'] for b in bits)
    max_x = max(b['x'] + b['w'] for b in bits)
    max_y = max(b['y'] + b['h'] for b in bits)
    content_width = max_x - min_x
    content_height = max_y - min_y
    orig_width, orig_height = original_image_size
    blocks = [{'x0': b['x'], 'y0': b['y'], 'x1': b['x'] + b['w'], 'y1': b['y'] + b['h']} for b in bits]
    margins = calculate_margins(blocks, orig_width, orig_height)
    left, top, right, bottom = margins['left'], margins['top'], margins['right'], margins['bottom']
    canvas_width = int(content_width + left + right)
    canvas_height = int(content_height + top + bottom)
    canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font_path = '/Library/Fonts/Courier New.ttf'
        base_font = ImageFont.truetype(font_path, 12)
    except:
        base_font = ImageFont.load_default()
    idx = 0
    full_text = decoded_text * 1000
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
        for size in range(10, max_font_size):
            try:
                font = ImageFont.truetype(font_path, size)
            except:
                font = base_font
            bbox = font.getbbox(line_text)
            text_width = bbox[2] - bbox[0]
            if text_width > line_width * 0.95:
                font_size = size - 1
                break
            font_size = size
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
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
    canvas.save(output_file)

def process_basic(image_path, config, decoded_text):
    color_img = cv2.imread(image_path)
    if color_img is None:
        print(f"[ERROR] Could not load image: {image_path}")
        return
    thresh_img = np.zeros_like(color_img[...,0])  # Placeholder for thresholded image
    bits = extract_bits_with_positions(thresh_img, color_img)
    lines = group_bits_by_lines(bits)
    original_image_size = (color_img.shape[1], color_img.shape[0])
    render_text_image(lines, bits, decoded_text, original_image_size, output_file="output/final_output_basic.jpeg") 