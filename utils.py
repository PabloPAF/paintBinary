"""
Shared utility functions for paintBinary project.
"""
import re
import os
import random
import cv2
import numpy as np

def clean_binary_string(binary_string):
    binary_string = ''.join(c for c in binary_string if c in '01')
    if len(binary_string) % 8 != 0:
        pad_len = 8 - (len(binary_string) % 8)
        binary_string = binary_string.ljust(len(binary_string) + pad_len, '0')
    return binary_string

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

def get_all_system_fonts():
    """
    Returns a list of all .ttf and .otf font file paths from common system font directories.
    """
    font_dirs = [
        "/Library/Fonts", "/System/Library/Fonts", os.path.expanduser("~/Library/Fonts"),
        "/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts")
    ]
    font_paths = set()
    for font_dir in font_dirs:
        if os.path.isdir(font_dir):
            for ext in ("ttf", "otf", "TTF", "OTF"):
                for path in os.listdir(font_dir):
                    if path.lower().endswith(ext.lower()):
                        font_paths.add(os.path.join(font_dir, path))
    return list(font_paths)

def get_random_font_path():
    fonts = get_all_system_fonts()
    if not fonts:
        return None
    return random.choice(fonts)

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

def recreate_shape_from_bits(lines, text, min_space_threshold=15, space_threshold=50):
    output_lines = []
    idx = 0
    full_text = text * 1000
    for line in lines:
        line = sorted(line, key=lambda b: b['x'])
        line_str = ''
        prev_bit = None
        if len(line) > 1:
            avg_char_width = sum(bit['w'] for bit in line) / len(line)
        else:
            avg_char_width = 10
        for bit in line:
            if prev_bit:
                gap = bit['x'] - (prev_bit['x'] + prev_bit['w'])
                if gap > min_space_threshold:
                    space_count = gap // space_threshold
                    line_str += ' ' * space_count
            line_str += full_text[idx % len(full_text)]
            idx += 1
            prev_bit = bit
        output_lines.append(line_str)
    return '\n'.join(output_lines)

def extract_data_from_image(image_path):
    color_img = cv2.imread(image_path)
    if color_img is None:
        print(f"[ERROR] Could not load image: {image_path}")
        return {'binary_str': '', 'bits': [], 'color_img': None}
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 10)
    bits = extract_bits_with_positions(thresh, color_img)
    binary_str = ''.join(b['bit'] for b in bits if 'bit' in b)
    return {
        'binary_str': binary_str,
        'bits': bits,
        'color_img': color_img,
        'thresh_img': thresh
    }

def extract_all_image_features(image_path, config):
    import cv2
    import numpy as np
    import pytesseract
    from PIL import Image
    print("📥 Loading and robustly preprocessing image...")
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not load image: {image_path}")
        return {}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    # Upscale for better OCR
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
    # Bit extraction (robust)
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
    bits = extract_bits_with_positions(cleaned, resized_color_img)
    # Per-line grouping
    lines = group_bits_by_lines(bits)
    # Contour/Block extraction (with section fallback)
    def is_canvas_mostly_white(color_img, white_thresh=240, percent=0.9):
        white_pixels = np.all(color_img > white_thresh, axis=2)
        white_ratio = np.sum(white_pixels) / (color_img.shape[0] * color_img.shape[1])
        return white_ratio >= percent
    def extract_colored_contours_with_positions(color_img, min_area=100):
        hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 00, 0])
        upper = np.array([254, 254, 254])
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blocks = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < config.get('min_area', 50):
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            mask_cnt = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
            mean_color = cv2.mean(color_img, mask=mask_cnt)[:3]
            mean_color = tuple(int(c) for c in mean_color)
            blocks.append({'x': x, 'y': y, 'w': w, 'h': h, 'color': mean_color})
        return blocks
    def extract_contours_by_section(color_img, grid_size=4, min_area=100):
        h, w, _ = color_img.shape
        section_h = h // grid_size
        section_w = w // grid_size
        blocks = []
        for i in range(grid_size):
            for j in range(grid_size):
                y0, y1 = i * section_h, (i + 1) * section_h if i < grid_size - 1 else h
                x0, x1 = j * section_w, (j + 1) * section_w if j < grid_size - 1 else w
                section = color_img[y0:y1, x0:x1]
                bg_color = np.median(section.reshape(-1, 3), axis=0).astype(np.uint8)
                diff = np.abs(section.astype(np.int16) - bg_color.astype(np.int16))
                mask = np.any(diff > 30, axis=2).astype(np.uint8) * 255
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < config.get('min_area', 50):
                        continue
                    x, y, w_box, h_box = cv2.boundingRect(cnt)
                    x_full = x0 + x
                    y_full = y0 + y
                    mask_cnt = np.zeros(mask.shape, np.uint8)
                    cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
                    mean_color = cv2.mean(section, mask=mask_cnt)[:3]
                    mean_color = tuple(int(c) for c in mean_color)
                    blocks.append({'x': x_full, 'y': y_full, 'w': w_box, 'h': h_box, 'color': mean_color})
        return blocks
    if is_canvas_mostly_white(resized_color_img):
        blocks = extract_colored_contours_with_positions(resized_color_img, min_area=config.get('min_area', 50))
    else:
        blocks = extract_contours_by_section(resized_color_img, grid_size=4, min_area=config.get('min_area', 50))
    # Macroblock grouping (Posterlike)
    def color_distance(c1, c2):
        return np.linalg.norm(np.array(c1) - np.array(c2))
    def blocks_are_close(b1, b2, proximity_thresh):
        x1, y1, w1, h1 = b1['x'], b1['y'], b1['w'], b1['h']
        x2, y2, w2, h2 = b2['x'], b2['y'], b2['w'], b2['h']
        dx = max(x1 - (x2 + w2), x2 - (x1 + w1), 0)
        dy = max(y1 - (y2 + h2), y2 - (y1 + h1), 0)
        return (dx ** 2 + dy ** 2) ** 0.5 < proximity_thresh
    def group_blocks_to_macroblocks(blocks, color_thresh=10, proximity_thresh=40):
        macroblocks = []
        used = set()
        for i, b in enumerate(blocks):
            if i in used:
                continue
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
            color = tuple(int(np.mean([blk['color'][k] for blk in group_blocks])) for k in range(3))
            min_x = min(blk['x'] for blk in group_blocks)
            min_y = min(blk['y'] for blk in group_blocks)
            max_x = max(blk['x'] + blk['w'] for blk in group_blocks)
            max_y = max(blk['y'] + blk['h'] for blk in group_blocks)
            bbox = (min_x, min_y, max_x, max_y)
            macroblocks.append({'blocks': group_blocks, 'color': color, 'bbox': bbox})
        return macroblocks
    macroblocks = group_blocks_to_macroblocks(blocks, color_thresh=config.get('macroblock_color_thresh', 10), proximity_thresh=config.get('macroblock_proximity_thresh', 40))
    # Margins
    def calculate_margins(blocks, canvas_width, canvas_height):
        left_margin = min(b['x'] for b in blocks)
        top_margin = min(b['y'] for b in blocks)
        right_margin = canvas_width - max(b['x'] + b['w'] for b in blocks)
        bottom_margin = canvas_height - max(b['y'] + b['h'] for b in blocks)
        return {
            'left': left_margin,
            'top': top_margin,
            'right': right_margin,
            'bottom': bottom_margin
        }
    margins = calculate_margins(blocks, resized_color_img.shape[1], resized_color_img.shape[0])
    return {
        'color_img': resized_color_img,
        'thresh_img': cleaned,
        'bits': bits,
        'binary_str': ''.join(b['bit'] for b in bits),
        'lines': lines,
        'blocks': blocks,
        'macroblocks': macroblocks,
        'margins': margins
    }

def process_image_to_shape(image_path, fallback_text="VivaPalestina", expected_chars=None, config=None):
    if config is None:
        from config import CONFIG
        config = CONFIG
    features = extract_all_image_features(image_path, config)
    bits = features.get('bits', [])
    lines = features.get('lines', [])
    # --- Per-line logic (old, default) ---
    binary_str_line = ''.join(b['bit'] for line in lines for b in sorted(line, key=lambda b: b['x']))
    decoded_text_line = binary_to_text(binary_str_line)
    # Check if at least 10 [a-zA-Z] in first 100 chars
    import re
    if len(re.findall(r'[a-zA-Z]', decoded_text_line[:100])) >= 10:
        method = 'per-line (old logic)'
        binary_str = binary_str_line
        decoded_text = decoded_text_line
    else:
        # --- Fallback: global (y, x) sort logic ---
        bits_sorted = sorted(bits, key=lambda b: (b['y'], b['x']))
        binary_str = ''.join(b['bit'] for b in bits_sorted)
        decoded_text = binary_to_text(binary_str)
        method = 'global (y, x) sort (fallback)'
    if expected_chars is not None:
        binary_str = binary_str[:expected_chars * 8]
        decoded_text = binary_to_text(binary_str)
    print(f"[DEBUG] Extraction method used: {method}")
    print(f"[DEBUG] Sorted & truncated binary string: {binary_str}")
    print(f"[DEBUG] Sorted binary string length: {len(binary_str)}")
    ascii_bytes = []
    for i in range(0, len(binary_str), 8):
        byte = binary_str[i:i+8]
        if len(byte) == 8:
            try:
                ascii_val = int(byte, 2)
                ascii_bytes.append(ascii_val)
            except Exception:
                ascii_bytes.append(None)
    print(f"[DEBUG] ASCII byte values: {ascii_bytes}")
    print(f"[DEBUG] Decoded text: {repr(decoded_text)}")
    nonprintable = [c for c in decoded_text if not c.isprintable()]
    if nonprintable:
        print(f"[DEBUG] Non-printable characters found: {nonprintable}")
    if not decoded_text or not decoded_text.isprintable():
        print("⚠️ Binary decoding failed or non-printable. Using fallback text.")
        decoded_text = fallback_text
    shape_output = recreate_shape_from_bits(lines, decoded_text)
    return decoded_text, shape_output, features 