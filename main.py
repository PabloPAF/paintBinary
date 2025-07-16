import cv2
import pytesseract
import numpy as np
import svgwrite
import re
import math
from PIL import Image, ImageDraw, ImageFont
from pytesseract import Output
import random
import os
from glob import glob

# Path to Tesseract (adjust if needed)
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'


def clean_binary_string(binary_string):
    # Strip any non-binary characters
    binary_string = ''.join(c for c in binary_string if c in '01')

    # Pad with zeros to make it a multiple of 8
    if len(binary_string) % 8 != 0:
        pad_len = 8 - (len(binary_string) % 8)
        binary_string = binary_string.ljust(len(binary_string) + pad_len, '0')

    return binary_string


def recreate_shape_from_bits_grid(bits, text):
    print("🧱 Reconstructing full grid shape...")

    if not bits:
        return ""

    # Step 1: Determine unique grid positions
    ys = sorted({b['y'] for b in bits})
    xs = sorted({b['x'] for b in bits})

    y_to_row = {y: i for i, y in enumerate(ys)}
    x_to_col = {x: i for i, x in enumerate(xs)}

    rows, cols = len(ys), len(xs)

    # Step 2: Create grid
    grid = [[None for _ in range(cols)] for _ in range(rows)]
    for b in bits:
        grid[y_to_row[b['y']]][x_to_col[b['x']]] = b

    it = iter((c for c in (text * 1000)))
    for r in range(rows):
        for c in range(cols):
            if grid[r][c]:
                grid[r][c]['char'] = next(it)

    output = "\n".join(
        "".join(grid[r][c]['char'] if grid[r][c] else " " for c in range(cols))
        for r in range(rows)
    )
    print(f"🧩 Completed grid with {rows} rows × {cols} cols")
    return output

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

def get_cropped_bounds_with_margins(blocks, margin_dict, canvas_width, canvas_height):
    """
    Compute a cropping box that applies the margins to the content bounds.

    Returns a bounding box tuple: (x0, y0, x1, y1)
    """
    if not blocks:
        return (0, 0, canvas_width, canvas_height)

    x0 = max(0, min(b['x0'] for b in blocks) - margin_dict['left'])
    y0 = max(0, min(b['y0'] for b in blocks) - margin_dict['top'])
    x1 = min(canvas_width, max(b['x1'] for b in blocks) + margin_dict['right'])
    y1 = min(canvas_height, max(b['y1'] for b in blocks) + margin_dict['bottom'])

    return (x0, y0, x1, y1)


from PIL import Image

def process_layout_with_margins(blocks, image):
    canvas_width, canvas_height = image.size

    # Step 1: Calculate margins
    margins = calculate_margins(blocks, canvas_width, canvas_height)

    # Step 2: Get crop box with applied margins
    crop_box = get_cropped_bounds_with_margins(blocks, margins, canvas_width, canvas_height)

    # Step 3: Crop the image
    cropped_image = image.crop(crop_box)

    # Step 4: Shift block coordinates relative to new origin
    # NOTE: Use crop_box x0/y0 instead of margins for correct delta
    dx = crop_box[0]
    dy = crop_box[1]

    adjusted_blocks = []
    for b in blocks:
        adjusted_block = {
            'x0': b['x0'] - dx,
            'y0': b['y0'] - dy,
            'x1': b['x1'] - dx,
            'y1': b['y1'] - dy,
            **{k: v for k, v in b.items() if k not in ['x0', 'y0', 'x1', 'y1']}
        }
        adjusted_blocks.append(adjusted_block)

    return cropped_image, adjusted_blocks


def render_text_image(lines, bits, decoded_text, original_image_size, output_file='output/translated_preview.png'):
    print("🖼️ Rendering image preview...")
    if not bits:
        print("⚠️ No bits to render.")
        return

    # Calculate content bounds
    min_x = min(b['x'] for b in bits)
    min_y = min(b['y'] for b in bits)
    max_x = max(b['x'] + b['w'] for b in bits)
    max_y = max(b['y'] + b['h'] for b in bits)
    content_width = max_x - min_x
    content_height = max_y - min_y

    # Calculate margins based on original image size
    orig_width, orig_height = original_image_size
    blocks = [{'x0': b['x'], 'y0': b['y'], 'x1': b['x'] + b['w'], 'y1': b['y'] + b['h']} for b in bits]
    margins = calculate_margins(blocks, orig_width, orig_height)
    left, top, right, bottom = margins['left'], margins['top'], margins['right'], margins['bottom']

    # Set canvas size to content + margins
    canvas_width = int(content_width + left + right)
    canvas_height = int(content_height + top + bottom)
    print(f"🧱 Canvas size: {canvas_width}×{canvas_height} (content: {content_width}×{content_height}, margins: L{left} T{top} R{right} B{bottom})")

    canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        font_path = "/Library/Fonts/Arial Unicode.ttf"  # Mac path
        base_font = ImageFont.truetype(font_path, 12)
    except:
        base_font = ImageFont.load_default()
        print("⚠️ Could not load custom font, using default.")

    idx = 0
    full_text = decoded_text * 1000
    print(f"🧾 Drawing {len(bits)} bits on a {canvas_width}×{canvas_height} image")
    print(f"🎯 Total lines to render: {len(lines)}")

    for line in lines:
        if not line:
            continue
        # Get the text for this line
        line_text = ''
        for _ in line:
            if idx >= len(full_text):
                break
            line_text += full_text[idx]
            idx += 1
        if not line_text:
            continue
        # Calculate available width
        x_positions = [bit['x'] for bit in line]
        min_line_x = min(x_positions)
        max_line_x = max(x_positions)
        line_width = max_line_x - min_line_x if len(x_positions) > 1 else 20
        # Dynamically find the best font size
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
        # Draw each character at its bit position, shifted by (min_x, min_y) and offset by margins
        for i, bit in enumerate(line):
            x = int(bit['x'] - min_x + left)
            y = int(bit['y'] - min_y + top)
            bgr = bit.get('color', (0, 0, 0))
            rgb = (bgr[2], bgr[1], bgr[0])
            color = rgb
            char = line_text[i] if i < len(line_text) else ' '
            if 0 <= x < canvas_width and 0 <= y < canvas_height:
                draw.text((x, y), char, fill=color, font=font)

    canvas.show()  # Opens image in preview window (macOS)
    canvas.save(output_file)
    print(f"✅ Image preview saved as {output_file}")


def apply_margins_to_blocks(blocks, margin_dict):
    """
    Adjust block coordinates so they are relative to the new top-left after margins are applied.
    Typically used after cropping.
    """
    adjusted_blocks = []
    for b in blocks:
        adjusted_block = {
            'x0': b['x0'] - margin_dict['left'],
            'y0': b['y0'] - margin_dict['top'],
            'x1': b['x1'] - margin_dict['left'],
            'y1': b['y1'] - margin_dict['top'],
            **{k: v for k, v in b.items() if k not in ['x0', 'y0', 'x1', 'y1']}  # Keep other metadata
        }
        adjusted_blocks.append(adjusted_block)

    return adjusted_blocks


def preprocess_image(image_path):
    print("📥 Loading and preprocessing image...")
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Resize for better OCR
    resized = cv2.resize(gray, None, fx=5, fy=5, interpolation=cv2.INTER_LINEAR)

    # Thresholding
    thresh = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 10)

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.dilate(cleaned, kernel, iterations=1)
    cv2.imwrite("output/debug_thresh.png", cleaned)
    print("✅ Preprocessing complete.")
    # Resize color_img to match resized gray shape
    resized_color_img = cv2.resize(img, (resized.shape[1], resized.shape[0]), interpolation=cv2.INTER_CUBIC)
    print(resized_color_img.shape)
    return cleaned, resized, resized_color_img


def extract_bits_with_positions(thresh_img, color_img):
    print("🔍 Detecting bits with OCR...")
    config = r'--oem 3 --psm 11 -c user_defined_dpi=300 tessedit_char_whitelist=01'

    h_img, w_img = thresh_img.shape
    boxes = pytesseract.image_to_boxes(thresh_img, config=config)
    print("📦 Number of raw OCR boxes:\n", len(boxes))

    bits = []

    for b in boxes.splitlines():
        try:
            ch, x1, y1, x2, y2, _ = b.split()
            if ch not in ['0', '1']:
                continue

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Tesseract uses bottom-left origin, OpenCV uses top-left
            y1 = h_img - y1
            y2 = h_img - y2

            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)

            # Make sure center is in image bounds
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

    print(f"✅ Extracted {len(bits)} valid bits.")
    return bits


def group_bits_by_lines(bits, band_height=20):
    print("📏 Grouping bits into horizontal bands...")

    if not bits:
        return []

    # Bucket bits by quantized y position
    bands = {}
    for bit in bits:
        band_key = (bit['y'] // band_height) * band_height
        bands.setdefault(band_key, []).append(bit)

    # Sort bands top-down
    sorted_band_keys = sorted(bands.keys())
    lines = []

    for key in sorted_band_keys:
        line = sorted(bands[key], key=lambda b: b['x'])  # left-to-right within band
        lines.append(line)

    print(f"📚 {len(lines)} lines formed using band height {band_height}.")
    return lines


def binary_to_text(binary_str):
    print("🔡 Decoding binary to ASCII...")
    chars = []
    binary = clean_binary_string(binary_str)

    for i in range(0, len(binary), 8):
        byte = binary[i:i + 8]
        if len(byte) == 8:
            try:
                char = chr(int(byte, 2))
                chars.append(char if char.isprintable() else '?')
            except ValueError:
                chars.append('?')

    decoded = ''.join(chars)
    print("🔠 Decoded Text (first 100 chars):")
    print(decoded[:100])

    # Extract the longest sequence of printable characters with reasonable length
    print("🔍 Extracting printable region...")
    matches = re.findall(r'[a-zA-Z0-9 .,%:;!?\'\"@#\(\)\[\]\{\}\-\n]{5,}', decoded)
    if matches:
        selected = max(matches, key=len)
        print("✅ Found valid ASCII segment:")
        print(selected)
        return selected
    else:
        print("⚠️ No valid ASCII text segment found. Using fallback.")
        return " "


def recreate_shape_from_bits(lines, text, min_space_threshold=15,space_threshold = 50):
    print("🧱 Reconstructing shape with string...")
    output_lines = []
    idx = 0
    full_text = text * 1000  # repeat as needed

    for line in lines:
        line = sorted(line, key=lambda b: b['x'])  # ensure left-to-right order
        line_str = ''
        prev_bit = None

        # Estimate average character width for this line
        if len(line) > 1:
            avg_char_width = sum(bit['w'] for bit in line) / len(line)
        else:
            avg_char_width = 10  # fallback default

        for bit in line:
            if prev_bit:
                gap = bit['x'] - (prev_bit['x'] + prev_bit['w'])  # from end of previous bit
                if gap > min_space_threshold:
                    # Compute number of spaces based on gap size
                    space_count = gap // space_threshold
                    line_str += ' ' * space_count

            line_str += full_text[idx % len(full_text)]
            idx += 1
            prev_bit = bit

        output_lines.append(line_str)

    print(f"🧩 Filled {len(output_lines)} lines using clean text.")
    return '\n'.join(output_lines)


from PIL import Image, ImageDraw

def reconstruct_grid(bit_string, rows, cols):
    total_required = rows * cols
    padded = bit_string.ljust(total_required, '0')  # pad if too short
    return padded[:total_required], rows, cols


def draw_bits_image(bits, rows, cols, block_size=90, output_path="output/translated_preview.png"):
    width = cols * block_size
    height = rows * block_size
    print(f"🖼️ Rendering image preview with size {width}×{height}")

    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    for i, bit in enumerate(bits):
        if bit == "1":
            row = i // cols
            col = i % cols
            x0 = col * block_size
            y0 = row * block_size
            x1 = x0 + block_size
            y1 = y0 + block_size
            draw.rectangle([x0, y0, x1, y1], fill="black")

    img.save(output_path)
    print(f"✅ Image saved to {output_path}")

def export_svg(bits, decoded_text, img_shape, filename='output/translated.svg'):
    print("🖋️ Generating SVG output...")
    height, width, _ = img_shape.shape
    dwg = svgwrite.Drawing(filename, size=(width, height), debug=True)

    font_size = 12
    for i, bit in enumerate(bits):
        if i >= len(decoded_text):
            break
        char = decoded_text[i]
        x, y, w, h = bit['x'], bit['y'], bit['w'], bit['h']
        r, g, b = bit['color'][2], bit['color'][1], bit['color'][0]  # BGR to RGB

        # Draw bounding box
        dwg.add(dwg.rect(insert=(x, y), size=(w, h),
                         fill='none', stroke='black', stroke_width=0.5))

        # Add character
        dwg.add(dwg.text(char,
                         insert=(x, y + h),
                         fill=svgwrite.rgb(r, g, b),
                         font_size=font_size,
                         font_family="Courier New"))  # Monospaced font

    dwg.save()

    # Save raw ASCII output
    with open("output/translated.txt", "w", encoding="utf-8", errors="replace") as f:
        f.write(decoded_text)

    print(f"✅ SVG saved to {filename}")

# Optional ASCII preview (console)
def print_ascii_grid(bits, rows, cols):
    for r in range(rows):
        line = ''.join(bits[r * cols + c] for c in range(cols))
        print(line)


def extract_colored_contours_with_positions(color_img, min_area=100):
    """
    Detect colored regions (contours) in the color image, extract their bounding boxes and average color.
    Returns a list of dicts: {'x', 'y', 'w', 'h', 'color'}
    """
    print("🔍 Detecting colored contours...")
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
    # Mask out near-white (background)
    lower = np.array([0, 00, 0])
    upper = np.array([254, 254, 254])  # V < 220 to exclude white
    mask = cv2.inRange(hsv, lower, upper)
    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bits = []
    debug_img = color_img.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # Compute average color inside the contour
        mask_cnt = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
        mean_color = cv2.mean(color_img, mask=mask_cnt)[:3]  # BGR
        mean_color = tuple(int(c) for c in mean_color)
        bits.append({'x': x, 'y': y, 'w': w, 'h': h, 'color': mean_color})
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imwrite('output/debug_colored_contours.png', debug_img)
    print(f"✅ Found {len(bits)} colored regions. Saved debug image: output/debug_colored_contours.png")
    return bits


def is_canvas_mostly_white(color_img, white_thresh=240, percent=0.9):
    """
    Returns True if at least `percent` of the image pixels are above `white_thresh` in all channels.
    """
    white_pixels = np.all(color_img > white_thresh, axis=2)
    white_ratio = np.sum(white_pixels) / (color_img.shape[0] * color_img.shape[1])
    print(f"🧪 White pixel ratio: {white_ratio:.2%}")
    return white_ratio >= percent


def extract_contours_by_section(color_img, grid_size=4, min_area=100):
    """
    Divide the image into sections, estimate the background color for each section, and extract contours using local background masking.
    Returns a list of dicts: {'x', 'y', 'w', 'h', 'color'}
    """
    print(f"🔍 Extracting contours by section (grid {grid_size}x{grid_size})...")
    h, w, _ = color_img.shape
    section_h = h // grid_size
    section_w = w // grid_size
    bits = []
    debug_img = color_img.copy()
    for i in range(grid_size):
        for j in range(grid_size):
            y0, y1 = i * section_h, (i + 1) * section_h if i < grid_size - 1 else h
            x0, x1 = j * section_w, (j + 1) * section_w if j < grid_size - 1 else w
            section = color_img[y0:y1, x0:x1]
            # Estimate background color as the mode of the section (or mean for robustness)
            bg_color = np.median(section.reshape(-1, 3), axis=0).astype(np.uint8)
            # Create mask for non-background (tolerant to small variations)
            diff = np.abs(section.astype(np.int16) - bg_color.astype(np.int16))
            mask = np.any(diff > 30, axis=2).astype(np.uint8) * 255
            # Morphological cleanup
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            # Find contours in this section
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area:
                    continue
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                # Offset by section position
                x_full = x0 + x
                y_full = y0 + y
                # Compute average color inside the contour
                mask_cnt = np.zeros(mask.shape, np.uint8)
                cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
                mean_color = cv2.mean(section, mask=mask_cnt)[:3]
                mean_color = tuple(int(c) for c in mean_color)
                bits.append({'x': x_full, 'y': y_full, 'w': w_box, 'h': h_box, 'color': mean_color})
                cv2.rectangle(debug_img, (x_full, y_full), (x_full + w_box, y_full + h_box), (0, 255, 0), 2)
    cv2.imwrite('output/debug_colored_contours_sections.png', debug_img)
    print(f"✅ Found {len(bits)} colored regions (by section). Saved debug image: output/debug_colored_contours_sections.png")
    return bits


def safe_extract_colored_contours_with_positions(color_img, min_area=100, white_thresh=240, percent=0.9, grid_size=4):
    """
    Safely extract colored contours: if the canvas is mostly white, use the standard method; otherwise, use section-based background extraction.
    """
    if is_canvas_mostly_white(color_img, white_thresh=white_thresh, percent=percent):
        print("✅ Canvas is mostly white. Using standard contour extraction.")
        return extract_colored_contours_with_positions(color_img, min_area=min_area)
    else:
        print("⚠️ Canvas is NOT mostly white. Using section-based background extraction.")
        return extract_contours_by_section(color_img, grid_size=grid_size, min_area=min_area)


def render_text_in_blocks(blocks, text, original_image_size, output_file='output/translated_blocks.png'):
    """
    For each detected color block, fill it with a portion of the extracted text, adjusting font and size to maximize fill and readability.
    """
    if not blocks:
        print("⚠️ No blocks to render.")
        return
    # Calculate content bounds
    min_x = int(min(b['x'] for b in blocks))
    min_y = int(min(b['y'] for b in blocks))
    max_x = int(max(b['x'] + b['w'] for b in blocks))
    max_y = int(max(b['y'] + b['h'] for b in blocks))
    content_width = max_x - min_x
    content_height = max_y - min_y
    # Calculate margins based on original image size
    orig_width, orig_height = original_image_size
    margin_dict = calculate_margins(
        [{'x0': b['x'], 'y0': b['y'], 'x1': b['x'] + b['w'], 'y1': b['y'] + b['h']} for b in blocks],
        orig_width, orig_height)
    left, top, right, bottom = margin_dict['left'], margin_dict['top'], margin_dict['right'], margin_dict['bottom']
    canvas_width = int(content_width + left + right)
    canvas_height = int(content_height + top + bottom)
    canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font_path = "/Library/Fonts/Arial Unicode.ttf"
        base_font = ImageFont.truetype(font_path, 12)
    except:
        base_font = ImageFont.load_default()
    idx = 0
    full_text = text * 1000
    for block in blocks:
        x = int(block['x'] - min_x + left)
        y = int(block['y'] - min_y + top)
        w = int(block['w'])
        h = int(block['h'])
        color = tuple(int(c) for c in block['color']) if isinstance(block['color'], (tuple, list)) else (0, 0, 0)
        # Assign a substring of text for this block
        chars_in_block = max(1, (w * h) // 600)  # heuristic: 1 char per ~600 px
        block_text = full_text[idx:idx+chars_in_block]
        idx += chars_in_block
        if not block_text:
            continue
        # Find max font size that fits
        font_size = 10
        max_font_size = min(w, h)
        for size in range(10, max_font_size):
            try:
                font = ImageFont.truetype(font_path, size)
            except:
                font = base_font
            bbox = font.getbbox(block_text)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            if text_width > w * 0.95 or text_height > h * 0.95:
                font_size = size - 1
                break
            font_size = size
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = base_font
        bbox = font.getbbox(block_text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        # Center text in block
        text_x = int(x + (w - text_width) // 2)
        text_y = int(y + (h - text_height) // 2)
        rgb = (color[2], color[1], color[0]) if len(color) == 3 else (0, 0, 0)
        draw.text((text_x, text_y), block_text, fill=rgb, font=font)
        # Optionally, draw block border for debug
        draw.rectangle([x, y, x + w, y + h], outline=(0, 0, 0), width=1)
    canvas.save(output_file)
    print(f"✅ Block text image saved as {output_file}")


def color_distance(c1, c2):
    return np.linalg.norm(np.array(c1) - np.array(c2))

def blocks_are_close(b1, b2, proximity_thresh):
    # Returns True if b1 and b2 are within proximity_thresh pixels of each other (bounding box distance)
    x1, y1, w1, h1 = b1['x'], b1['y'], b1['w'], b1['h']
    x2, y2, w2, h2 = b2['x'], b2['y'], b2['w'], b2['h']
    # Compute the closest distance between the two rectangles
    dx = max(x1 - (x2 + w2), x2 - (x1 + w1), 0)
    dy = max(y1 - (y2 + h2), y2 - (y1 + h1), 0)
    return (dx ** 2 + dy ** 2) ** 0.5 < proximity_thresh

def group_blocks_to_macroblocks(blocks, color_thresh=10, proximity_thresh=40):
    """
    Group blocks into macroblocks by color similarity and proximity.
    Returns a list of macroblocks, each as a dict with keys: 'blocks', 'color', 'bbox'.
    """
    macroblocks = []
    used = set()
    for i, b in enumerate(blocks):
        if i in used:
            continue
        # Start a new macroblock
        group = [i]
        queue = [i]
        while queue:
            idx = queue.pop()
            for j, other in enumerate(blocks):
                if j in used or j in group:
                    continue
                if color_distance(b['color'], other['color']) < color_thresh and blocks_are_close(blocks[idx], other, proximity_thresh):
                    group.append(j)
                    queue.append(j)
        for idx in group:
            used.add(idx)
        group_blocks = [blocks[idx] for idx in group]
        # Macroblock color: mean color of all blocks
        color = tuple(int(np.mean([blk['color'][k] for blk in group_blocks])) for k in range(3))
        # Macroblock bbox: min/max of all blocks
        min_x = min(blk['x'] for blk in group_blocks)
        min_y = min(blk['y'] for blk in group_blocks)
        max_x = max(blk['x'] + blk['w'] for blk in group_blocks)
        max_y = max(blk['y'] + blk['h'] for blk in group_blocks)
        bbox = (min_x, min_y, max_x, max_y)
        macroblocks.append({'blocks': group_blocks, 'color': color, 'bbox': bbox})
    return macroblocks


def merge_macroblocks(macroblocks, proximity_thresh=20, color_thresh=5):
    """
    Merge macroblocks that are within proximity_thresh and have color distance < color_thresh.
    Returns a new list of merged macroblocks. Logs the number of merges performed.
    """
    import numpy as np
    merged = []
    used = set()
    merges = 0
    for i, m1 in enumerate(macroblocks):
        if i in used:
            continue
        group = [i]
        queue = [i]
        while queue:
            idx = queue.pop()
            m_a = macroblocks[idx]
            for j, m_b in enumerate(macroblocks):
                if j in used or j in group:
                    continue
                # Proximity: bounding boxes within threshold
                a_min_x, a_min_y, a_max_x, a_max_y = m_a['bbox']
                b_min_x, b_min_y, b_max_x, b_max_y = m_b['bbox']
                dx = max(a_min_x - b_max_x, b_min_x - a_max_x, 0)
                dy = max(a_min_y - b_max_y, b_min_y - a_max_y, 0)
                if (dx ** 2 + dy ** 2) ** 0.5 > proximity_thresh:
                    continue
                # Color distance
                c1 = np.array(m_a['color'])
                c2 = np.array(m_b['color'])
                if np.linalg.norm(c1 - c2) > color_thresh:
                    continue
                group.append(j)
                queue.append(j)
                merges += 1
        for idx in group:
            used.add(idx)
        group_blocks = [macroblocks[idx] for idx in group]
        # Merge bbox
        min_x = min(m['bbox'][0] for m in group_blocks)
        min_y = min(m['bbox'][1] for m in group_blocks)
        max_x = max(m['bbox'][2] for m in group_blocks)
        max_y = max(m['bbox'][3] for m in group_blocks)
        # Merge color (average)
        color = tuple(int(np.mean([m['color'][k] for m in group_blocks])) for k in range(3))
        merged.append({'blocks': sum([m['blocks'] for m in group_blocks], []), 'color': color, 'bbox': (min_x, min_y, max_x, max_y)})
    print(f"Macroblock merges performed: {merges}")
    return merged


def get_system_fonts():
    # Try to get a list of system font files (ttf/otf)
    font_paths = set()
    for font_dir in ["/Library/Fonts", "/System/Library/Fonts", os.path.expanduser("~/Library/Fonts")]:
        if os.path.isdir(font_dir):
            for ext in ("ttf", "otf", "TTF", "OTF"):
                font_paths.update(glob(os.path.join(font_dir, f"*.{ext}")))
    return list(font_paths)


def find_font_variants(font_path):
    # Try to find bold/italic variants in the same directory as the font_path
    base_dir = os.path.dirname(font_path)
    base_name = os.path.splitext(os.path.basename(font_path))[0].lower()
    variants = {'regular': font_path, 'bold': None, 'italic': None, 'bolditalic': None}
    for f in os.listdir(base_dir):
        f_lower = f.lower()
        if base_name in f_lower:
            if 'bold' in f_lower and 'italic' in f_lower:
                variants['bolditalic'] = os.path.join(base_dir, f)
            elif 'bold' in f_lower:
                variants['bold'] = os.path.join(base_dir, f)
            elif 'italic' in f_lower or 'oblique' in f_lower:
                variants['italic'] = os.path.join(base_dir, f)
    return variants


def wrap_text_to_fit(draw, text, font, max_width):
    # Wrap text into lines so that each line fits within max_width
    words = text.split()
    lines = []
    current_line = ''
    for word in words:
        test_line = current_line + (' ' if current_line else '') + word
        bbox = font.getbbox(test_line)
        if bbox[2] - bbox[0] <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines


def wrap_text_to_fit_char(draw, text, font, max_width):
    # Wrap text into lines so that each line fits within max_width, breaking at any character
    lines = []
    current_line = ''
    for char in text:
        test_line = current_line + char
        bbox = font.getbbox(test_line)
        if bbox[2] - bbox[0] <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = char
    if current_line:
        lines.append(current_line)
    return lines


def render_text_in_macroblocks(macroblocks, text, original_image_size, output_file='output/translated_macroblocks.png'):
    if not macroblocks:
        print("⚠️ No macroblocks to render.")
        return
    orig_width, orig_height = original_image_size
    max_dim = 4096
    scale = 1.0
    if orig_width > max_dim or orig_height > max_dim:
        scale = min(max_dim / orig_width, max_dim / orig_height)
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        print(f"⚠️ Downscaling output canvas from {orig_width}x{orig_height} to {new_width}x{new_height}")
    else:
        new_width, new_height = orig_width, orig_height
    if new_width > 10000 or new_height > 10000:
        print(f"❌ Image size {new_width}x{new_height} is too large, skipping macroblock rendering.")
        return
    collage_layer = Image.new('RGBA', (new_width, new_height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(collage_layer)
    debug_img = Image.new('RGB', (new_width, new_height), (255, 255, 255))
    debug_draw = ImageDraw.Draw(debug_img)
    fonts = get_system_fonts()
    if not fonts:
        fonts = [None]
    idx = 0
    full_text = text * 10000
    min_block_size = 100
    for macro in macroblocks:
        min_x, min_y, max_x, max_y = macro['bbox']
        min_x = int(max(0, min(min_x, orig_width - 1)) * scale)
        min_y = int(max(0, min(min_y, orig_height - 1)) * scale)
        max_x = int(max(0, min(max_x, orig_width)) * scale)
        max_y = int(max(0, min(max_y, orig_height)) * scale)
        w = max_x - min_x
        h = max_y - min_y
        # Skip macroblocks that are too small or invalid
        if w <= 0 or h <= 0:
            print(f"⚠️ Skipping macroblock with invalid size: ({w}x{h}) at ({min_x},{min_y})")
            continue
        color = macro['color']
        block_x0 = min_x
        block_y0 = min_y
        block_x1 = max_x
        block_y1 = max_y
        block_w = max(1, block_x1 - block_x0)
        block_h = max(1, block_y1 - block_y0)
        min_font = int(0.8 * min(block_w, block_h))
        max_font = int(2.5 * min(block_w, block_h))
        step = max(1, min_font // 5)
        n_chars = int((block_w * block_h) // (step * step)) * 5
        if n_chars < 1:
            n_chars = 1
        for i in range(n_chars):
            char = full_text[(idx + i) % len(full_text)]
            font_path = random.choice(fonts)
            variants = find_font_variants(font_path) if font_path else {'regular': None}
            if variants.get('bold'):
                font_file = variants['bold']
            elif variants.get('bolditalic'):
                font_file = variants['bolditalic']
            else:
                font_file = variants.get('regular') or font_path
            font_size = random.randint(min_font, max_font)
            try:
                font = ImageFont.truetype(font_file, font_size) if font_file else ImageFont.load_default()
            except:
                font = ImageFont.load_default()
            angle = random.uniform(-50, 50)
            bbox = font.getbbox(char)
            char_w = bbox[2] - bbox[0]
            char_h = bbox[3] - bbox[1]
            x = random.randint(int(block_x0 - char_w // 2), int(block_x1 - char_w // 2))
            y = random.randint(int(block_y0 - char_h // 2), int(block_y1 - char_h // 2))
            char_img = Image.new('RGBA', (int(char_w * 2), int(char_h * 2)), (255, 255, 255, 0))
            char_draw = ImageDraw.Draw(char_img)
            rgb = (color[2], color[1], color[0]) if len(color) == 3 else (0, 0, 0)
            char_draw.text((char_w // 2, char_h // 2), char, fill=rgb, font=font)
            try:
                rotated = char_img.rotate(angle, expand=1, resample=Image.Resampling.BICUBIC)
            except AttributeError:
                rotated = char_img.rotate(angle, expand=1)
            collage_layer.paste(rotated, (x, y), rotated)
        idx += n_chars
        debug_draw.rectangle([min_x, min_y, max_x, max_y], outline=(255, 0, 0), width=4)
    final_img = Image.new('RGB', (new_width, new_height), (255, 255, 255))
    final_img.paste(collage_layer, (0, 0), collage_layer)
    final_img.save(output_file)
    debug_img.save('output/debug_macroblocks.png')
    print(f"✅ Macroblock text image saved as {output_file}")
    print(f"✅ Macroblock coverage debug image saved as output/debug_macroblocks.png")


def render_macroblock_grid_preview(macroblocks, text, original_image_size, output_file='output/debug_colorBlocks.png'):
    """
    For each macroblock, fill its bounding box with a dense grid of text, using the macroblock's color.
    The style is regular grid (not collage/overlap), edge-to-edge, using a monospaced font if possible.
    This is for debugging: macroblock boundaries may be visible.
    """
    if not macroblocks:
        print("⚠️ No macroblocks to render.")
        return
    orig_width, orig_height = original_image_size
    canvas = Image.new('RGB', (orig_width, orig_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font_path = "/Library/Fonts/Courier New.ttf"
        base_font = ImageFont.truetype(font_path, 12)
    except:
        base_font = ImageFont.load_default()
        print("⚠️ Could not load monospaced font, using default.")
    idx = 0
    full_text = text * 10000
    for macro in macroblocks:
        min_x, min_y, max_x, max_y = macro['bbox']
        w = max_x - min_x
        h = max_y - min_y
        if w <= 0 or h <= 0:
            continue
        color = macro['color']
        rgb = (color[2], color[1], color[0]) if len(color) == 3 else (0, 0, 0)
        font_size = 10
        max_font_size = min(w, h)
        chars_across = 1
        chars_down = 1
        for size in range(8, max_font_size):
            try:
                font = ImageFont.truetype(font_path, size)
            except:
                font = base_font
            bbox = font.getbbox('M')
            char_w = bbox[2] - bbox[0]
            char_h = bbox[3] - bbox[1]
            n_across = max(1, w // char_w)
            n_down = max(1, h // char_h)
            if n_across * char_w > w * 0.98 or n_down * char_h > h * 0.98:
                font_size = size - 1
                break
            font_size = size
            chars_across = n_across
            chars_down = n_down
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = base_font
        bbox = font.getbbox('M')
        char_w = bbox[2] - bbox[0]
        char_h = bbox[3] - bbox[1]
        n_across = max(1, w // char_w)
        n_down = max(1, h // char_h)
        for row in range(n_down):
            for col in range(n_across):
                char = full_text[idx % len(full_text)]
                idx += 1
                x = int(min_x + col * char_w)
                y = int(min_y + row * char_h)
                if x + char_w <= max_x and y + char_h <= max_y:
                    draw.text((x, y), char, fill=rgb, font=font)
        # Draw macroblock boundary for debug
        draw.rectangle([min_x, min_y, max_x, max_y], outline=(255, 0, 0), width=2)
    canvas.save(output_file)
    print(f"✅ Macroblock grid debug preview saved as {output_file}")


def render_macroblock_grid_final(macroblocks, text, original_image_size, output_file):
    """
    For each macroblock, fill its bounding box with a dense grid of text, using black text for debugging.
    The style is regular grid (not collage/overlap), edge-to-edge, using a monospaced font if possible.
    This is the final output: NO macroblock outlines or debug overlays, and no colored fills.
    """
    if not macroblocks:
        print("⚠️ No macroblocks to render.")
        return
    orig_width, orig_height = original_image_size
    canvas = Image.new('RGB', (orig_width, orig_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font_path = "/Library/Fonts/Courier New.ttf"
        base_font = ImageFont.truetype(font_path, 20)
    except Exception as e:
        print(f"⚠️ Could not load monospaced font: {e}. Using default.")
        base_font = ImageFont.load_default()
    idx = 0
    full_text = text * 10000
    for macro_idx, macro in enumerate(macroblocks):
        min_x, min_y, max_x, max_y = macro['bbox']
        w = max_x - min_x
        h = max_y - min_y
        if w <= 0 or h <= 0:
            continue
        text_color = (0, 0, 0)  # Always black for debugging
        min_font_size = 20
        max_font_size = min(w, h)
        font_size = min_font_size
        chars_across = 1
        chars_down = 1
        for size in range(min_font_size, max_font_size):
            try:
                font = ImageFont.truetype(font_path, size)
            except:
                font = base_font
            bbox = font.getbbox('M')
            char_w = bbox[2] - bbox[0]
            char_h = bbox[3] - bbox[1]
            n_across = max(1, w // char_w)
            n_down = max(1, h // char_h)
            if n_across * char_w > w * 0.98 or n_down * char_h > h * 0.98:
                font_size = size - 1
                break
            font_size = size
            chars_across = n_across
            chars_down = n_down
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = base_font
        bbox = font.getbbox('M')
        char_w = bbox[2] - bbox[0]
        char_h = bbox[3] - bbox[1]
        n_across = max(1, w // char_w)
        n_down = max(1, h // char_h)
        print(f"Macroblock {macro_idx}: font_size={font_size}, chars_across={n_across}, chars_down={n_down}, char_w={char_w}, char_h={char_h}")
        for row in range(n_down):
            for col in range(n_across):
                char = full_text[idx % len(full_text)]
                idx += 1
                x = int(min_x + col * char_w)
                y = int(min_y + row * char_h)
                if x + char_w <= max_x and y + char_h <= max_y:
                    draw.text((x, y), char, fill=text_color, font=font)
                    if row == 0 and col == 0:
                        print(f"  Sample char: '{char}' at ({x},{y})")
    canvas.save(output_file)
    print(f"✅ Final macroblock grid output saved as {output_file}")




def render_text_in_macroblocks_gridstyle(macroblocks, text, original_image_size, output_file):
    """
    Fill macroblocks with multiple characters (grid approach). For large macroblocks, render a grid of characters.
    For small macroblocks, render one large character. Use progressive font size testing (250pt down to 11pt).
    Use the macroblock's color for the text. No outlines or debug overlays.
    """
    if not macroblocks:
        print("⚠️ No macroblocks to render.")
        return
    orig_width, orig_height = original_image_size
    canvas = Image.new('RGB', (orig_width, orig_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font_path = "/Library/Fonts/Courier New.ttf"
        base_font = ImageFont.truetype(font_path, 12)
    except Exception as e:
        print(f"⚠️ Could not load monospaced font: {e}. Using default.")
        base_font = ImageFont.load_default()
    idx = 0
    full_text = text * 10000
    min_font_pt = 11
    max_font_pt = 250
    min_area = 50
    large_block_threshold = 2000  # Area threshold to decide between grid vs single char
    skipped_area = 0
    skipped_font = 0
    total = len(macroblocks)
    font_sizes_used = []
    for i, macro in enumerate(macroblocks):
        min_x, min_y, max_x, max_y = macro['bbox']
        w = max_x - min_x
        h = max_y - min_y
        area = w * h
        if w <= 0 or h <= 0 or area < min_area:
            skipped_area += 1
            continue
        color = macro['color']
        text_color = (color[2], color[1], color[0]) if len(color) == 3 else (0, 0, 0)
        
        # Decide approach based on macroblock size
        if area >= large_block_threshold:
            # Large macroblock: use grid approach
            font_size = None
            best_grid = (1, 1)
            for size in range(max_font_pt, min_font_pt - 1, -1):
                try:
                    font = ImageFont.truetype(font_path, size)
                except:
                    font = base_font
                bbox = font.getbbox('M')
                char_w = bbox[2] - bbox[0]
                char_h = bbox[3] - bbox[1]
                if char_w <= w and char_h <= h:
                    n_across = max(1, w // char_w)
                    n_down = max(1, h // char_h)
                    if n_across * char_w <= w * 0.98 and n_down * char_h <= h * 0.98:
                        font_size = size
                        best_grid = (n_across, n_down)
                        break
            if font_size is None:
                skipped_font += 1
                continue
            try:
                font = ImageFont.truetype(font_path, font_size)
            except:
                font = base_font
            bbox = font.getbbox('M')
            char_w = bbox[2] - bbox[0]
            char_h = bbox[3] - bbox[1]
            n_across, n_down = best_grid
            for row in range(n_down):
                for col in range(n_across):
                    char = full_text[idx % len(full_text)]
                    idx += 1
                    x = int(min_x + col * char_w)
                    y = int(min_y + row * char_h)
                    if x + char_w <= max_x and y + char_h <= max_y:
                        draw.text((x, y), char, fill=text_color, font=font)
            print(f"Macroblock {i} (large): font_size={font_size}, grid={n_across}x{n_down}, bbox=({min_x},{min_y},{max_x},{max_y})")
        else:
            # Small macroblock: use single large character
            font_size = None
            for size in range(max_font_pt, min_font_pt - 1, -1):
                try:
                    font = ImageFont.truetype(font_path, size)
                except:
                    font = base_font
                bbox = font.getbbox('M')
                char_w = bbox[2] - bbox[0]
                char_h = bbox[3] - bbox[1]
                if char_w <= w and char_h <= h:
                    font_size = size
                    break
            if font_size is None:
                skipped_font += 1
                continue
            try:
                font = ImageFont.truetype(font_path, font_size)
            except:
                font = base_font
            bbox = font.getbbox('M')
            char_w = bbox[2] - bbox[0]
            char_h = bbox[3] - bbox[1]
            x = int(min_x + (w - char_w) // 2)
            y = int(min_y + (h - char_h) // 2)
            char = full_text[idx % len(full_text)]
            idx += 1
            draw.text((x, y), char, fill=text_color, font=font)
            print(f"Macroblock {i} (small): font_size={font_size}, char='{char}', bbox=({min_x},{min_y},{max_x},{max_y})")
        
        font_sizes_used.append(font_size)
    canvas.save(output_file)
    print(f"✅ Macroblock grid-style text output saved as {output_file}")
    print(f"Macroblocks: total={total}, skipped_area={skipped_area}, skipped_font={skipped_font}, rendered={total-skipped_area-skipped_font}")
    if font_sizes_used:
        print(f"Font sizes used: min={min(font_sizes_used)}, max={max(font_sizes_used)}, avg={sum(font_sizes_used)/len(font_sizes_used):.1f}")
        print(f"Biggest font size applied: {max(font_sizes_used)}pt")


def process_image_to_shape(image_path, fallback_text="VivaPalestina"):
    thresh_img, resized_img, color_img = preprocess_image(image_path)
    scale_x = color_img.shape[1] / thresh_img.shape[1]
    scale_y = color_img.shape[0] / thresh_img.shape[0]
    # 1. Try to extract decoded text using OCR-based bit extraction
    ocr_bits = extract_bits_with_positions(thresh_img, color_img)
    binary_str = ''.join(b['bit'] for b in ocr_bits)
    decoded_text = binary_to_text(binary_str)
    # 2. If decoded_text is empty or not printable, use fallback
    if not decoded_text or not decoded_text.isprintable():
        print("⚠️ Binary decoding failed or non-printable. Using fallback text.")
        decoded_text = fallback_text
    # 3. Extract colored regions (contours)
    bits = safe_extract_colored_contours_with_positions(color_img)
    for b in bits:
        b['bit'] = '1'  # not used for text, just for compatibility
    # Adjust coordinates
    for b in bits:
        b['x'] = int(b['x'] * scale_x)
        b['y'] = int(b['y'] * scale_y)
        b['w'] = int(b['w'] * scale_x)
        b['h'] = int(b['h'] * scale_y)
    cv2.imwrite("output/debug_thresh_check.png", thresh_img)
    if not bits:
        print("❌ No bits detected.")
        return None, None
    # 4. Use decoded_text to fill the regions
    shape_output = recreate_shape_from_bits_grid(bits, decoded_text)
    lines = group_bits_by_lines(bits)
    original_image_size = (color_img.shape[1], color_img.shape[0])
    macroblocks = group_blocks_to_macroblocks(bits)
    print(f"Macroblocks before merging: {len(macroblocks)}")
    merged_macroblocks = merge_macroblocks(macroblocks, proximity_thresh=120, color_thresh=20)
    print(f"Macroblocks after merging: {len(merged_macroblocks)}")
    # Debug preview with macroblock outlines
    render_macroblock_grid_preview(merged_macroblocks, decoded_text, original_image_size, output_file="output/debug_colorBlocks.png")
    # Final output: use merged macroblocks in grid style
    import os
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    final_output_path = f"output/final_output_{base_name}.jpeg"
    render_text_in_macroblocks_gridstyle(merged_macroblocks, decoded_text, original_image_size, output_file=final_output_path)
    render_text_in_blocks(bits, decoded_text, original_image_size, output_file="output/translated_blocks.png")
    render_text_in_macroblocks(merged_macroblocks, decoded_text, original_image_size, output_file="output/translated_macroblocks.png")
    export_svg(bits, decoded_text, color_img)
    print("\n🔠 Decoded Text:\n", decoded_text)
    print("\n🎩 Reconstructed Shape:\n", shape_output)
    return decoded_text, shape_output

if __name__ == '__main__':
    image_path = "test/FlagColor.jpeg"
    process_image_to_shape(image_path)
