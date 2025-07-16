"""
Posterlike module: Extracts colored regions and renders text in a collage/poster style.
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from utils import get_random_font_path, process_image_to_shape

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

def merge_macroblocks(macroblocks, proximity_thresh=20, color_thresh=5):
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
                a_min_x, a_min_y, a_max_x, a_max_y = m_a['bbox']
                b_min_x, b_min_y, b_max_x, b_max_y = m_b['bbox']
                dx = max(a_min_x - b_max_x, b_min_x - a_max_x, 0)
                dy = max(a_min_y - b_max_y, b_min_y - a_max_y, 0)
                if (dx ** 2 + dy ** 2) ** 0.5 > proximity_thresh:
                    continue
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
        min_x = min(m['bbox'][0] for m in group_blocks)
        min_y = min(m['bbox'][1] for m in group_blocks)
        max_x = max(m['bbox'][2] for m in group_blocks)
        max_y = max(m['bbox'][3] for m in group_blocks)
        color = tuple(int(np.mean([m['color'][k] for m in group_blocks])) for k in range(3))
        merged.append({'blocks': sum([m['blocks'] for m in group_blocks], []), 'color': color, 'bbox': (min_x, min_y, max_x, max_y)})
    print(f"Macroblock merges performed: {merges}")
    return merged

def safe_extract_colored_contours_with_positions(color_img, min_area=100, white_thresh=240, percent=0.9, grid_size=4):
    def is_canvas_mostly_white(color_img, white_thresh=240, percent=0.9):
        white_pixels = np.all(color_img > white_thresh, axis=2)
        white_ratio = np.sum(white_pixels) / (color_img.shape[0] * color_img.shape[1])
        print(f"🧪 White pixel ratio: {white_ratio:.2%}")
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
        bits = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            mask_cnt = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
            mean_color_raw = cv2.mean(color_img, mask=mask_cnt)
            if isinstance(mean_color_raw, (tuple, list)) and len(mean_color_raw) >= 3:
                mean_color = tuple(int(c) for c in mean_color_raw[:3])
            else:
                mean_color = (0, 0, 0)
            bits.append({'x': x, 'y': y, 'w': w, 'h': h, 'color': mean_color})
        return bits
    def extract_contours_by_section(color_img, grid_size=4, min_area=100):
        h, w, _ = color_img.shape
        section_h = h // grid_size
        section_w = w // grid_size
        bits = []
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
                    if area < min_area:
                        continue
                    x, y, w_box, h_box = cv2.boundingRect(cnt)
                    x_full = x0 + x
                    y_full = y0 + y
                    mask_cnt = np.zeros(mask.shape, np.uint8)
                    cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
                    mean_color_raw = cv2.mean(section, mask=mask_cnt)
                    if isinstance(mean_color_raw, (tuple, list)) and len(mean_color_raw) >= 3:
                        mean_color = tuple(int(c) for c in mean_color_raw[:3])
                    else:
                        mean_color = (0, 0, 0)
                    bits.append({'x': x_full, 'y': y_full, 'w': w_box, 'h': h_box, 'color': mean_color})
        return bits
    if is_canvas_mostly_white(color_img, white_thresh=white_thresh, percent=percent):
        print("✅ Canvas is mostly white. Using standard contour extraction.")
        return extract_colored_contours_with_positions(color_img, min_area=min_area)
    else:
        print("⚠️ Canvas is NOT mostly white. Using section-based background extraction.")
        return extract_contours_by_section(color_img, grid_size=grid_size, min_area=min_area)

def render_text_in_macroblocks_gridstyle(macroblocks, text, original_image_size, output_file, config=None):
    if not macroblocks:
        print("⚠️ No macroblocks to render.")
        return
    orig_width, orig_height = original_image_size
    canvas = Image.new('RGB', (orig_width, orig_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    idx = 0
    full_text = text * 10000
    min_font_pt = 11
    max_font_pt = 250
    min_area = 50
    large_block_threshold = 2000
    for i, macro in enumerate(macroblocks):
        min_x, min_y, max_x, max_y = macro['bbox']
        w = max_x - min_x
        h = max_y - min_y
        area = w * h
        if w <= 0 or h <= 0 or area < min_area:
            continue
        color = macro['color']
        text_color = (color[2], color[1], color[0]) if len(color) == 3 else (0, 0, 0)
        # Font selection logic
        if config and config.get('font_path'):
            font_path = config['font_path']
        else:
            font_path = get_random_font_path()
        # Decide approach based on macroblock size
        if area >= large_block_threshold:
            font_size = None
            best_grid = (1, 1)
            for size in range(max_font_pt, min_font_pt - 1, -1):
                if font_path:
                    try:
                        font = ImageFont.truetype(font_path, size)
                    except:
                        font = ImageFont.load_default()
                else:
                    font = ImageFont.load_default()
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
                continue
            if font_path:
                try:
                    font = ImageFont.truetype(font_path, font_size)
                except:
                    font = ImageFont.load_default()
            else:
                font = ImageFont.load_default()
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
        else:
            font_size = None
            for size in range(max_font_pt, min_font_pt - 1, -1):
                if font_path:
                    try:
                        font = ImageFont.truetype(font_path, size)
                    except:
                        font = ImageFont.load_default()
                else:
                    font = ImageFont.load_default()
                bbox = font.getbbox('M')
                char_w = bbox[2] - bbox[0]
                char_h = bbox[3] - bbox[1]
                if char_w <= w and char_h <= h:
                    font_size = size
                    break
            if font_size is None:
                continue
            if font_path:
                try:
                    font = ImageFont.truetype(font_path, font_size)
                except:
                    font = ImageFont.load_default()
            else:
                font = ImageFont.load_default()
            bbox = font.getbbox('M')
            char_w = bbox[2] - bbox[0]
            char_h = bbox[3] - bbox[1]
            x = int(min_x + (w - char_w) // 2)
            y = int(min_y + (h - char_h) // 2)
            char = full_text[idx % len(full_text)]
            idx += 1
            draw.text((x, y), char, fill=text_color, font=font)
    canvas.save(output_file)

def process_posterlike(image_path, config, decoded_text, features=None):
    # Use provided features or extract if not provided
    if features is None:
        decoded_text, shape_output, features = process_image_to_shape(image_path, config=config)
    macroblocks = features.get('macroblocks', [])
    original_image_size = (features['color_img'].shape[1], features['color_img'].shape[0]) if features.get('color_img') is not None else (0, 0)
    render_text_in_macroblocks_gridstyle(macroblocks, decoded_text, original_image_size, output_file="output/final_output_posterlike.jpeg", config=config) 