import cv2
import pytesseract
import numpy as np
import svgwrite
import re
import math
from PIL import Image, ImageDraw, ImageFont
from pytesseract import Output

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


def render_text_image(lines, bits, decoded_text, output_file='output/translated_preview.png'):
    print("🖼️ Rendering image preview...")
    if not bits:
        print("⚠️ No bits to render.")
        return

    min_x = min(b['x'] for b in bits)
    min_y = min(b['y'] for b in bits)
    max_x = max(b['x'] for b in bits)
    max_y = max(b['y'] for b in bits)
    canvas_width = max_x - min_x + 100
    canvas_height = max_y - min_y + 100
    print(f"🧱 Canvas size: {canvas_width}×{canvas_height}")

    canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))

    print("📍 Sample bits with position and color:")
    for bit in bits[:5]:
        print(f"  x={bit['x']}, y={bit['y']}, h={bit['h']}, color={bit['color']}")

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
        x_positions = [bit['x'] - min_x for bit in line]
        min_line_x = min(x_positions)
        max_line_x = max(x_positions)
        line_width = max_line_x - min_line_x if len(x_positions) > 1 else 20
        # Estimate average char width
        avg_char_width = line_width / max(1, len(line_text))
        # Dynamically find the best font size
        font_size = 10
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
        # Draw each character at its bit position
        for i, bit in enumerate(line):
            x = bit['x'] - min_x
            y = bit['y'] - min_y
            bgr = bit.get('color', (0, 0, 0))
            rgb = (bgr[2], bgr[1], bgr[0])
            color = rgb
            char = line_text[i] if i < len(line_text) else ' '
            if 0 <= x < canvas_width and 0 <= y < canvas_height:
                draw.text((x, y), char, fill=color, font=font)

    canvas.show()  # Opens image in preview window (macOS)
    canvas.save(output_file)
    print(f"✅ Image preview saved as {output_file}")


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
    print("📦 Raw OCR boxes:\n", boxes)

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
        return "Viva Palestina"


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


def draw_bits_image(bits, rows, cols, block_size=10, output_path="output/translated_preview.png"):
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
    lower = np.array([0, 0, 0])
    upper = np.array([180, 255, 220])  # V < 220 to exclude white
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


def process_image_to_shape(image_path, fallback_text="Viva Palestina"):
    thresh_img, resized_img, color_img = preprocess_image(image_path)
    scale_x = color_img.shape[1] / thresh_img.shape[1]
    scale_y = color_img.shape[0] / thresh_img.shape[0]
    # bits = extract_bits_with_positions(thresh_img, color_img)
    # --- To use contour-based extraction instead of OCR, comment the above and uncomment below ---
    bits = extract_colored_contours_with_positions(color_img)
    for b in bits:
        b['bit'] = '1'  # or assign text as needed

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

    binary_str = ''.join(b.get('bit', '1') for b in bits)
    print(f"🧮 Binary string: {binary_str[:64]}...")
    # === [STEP 4] Render grid from bit string ===
    # Let's assume we manually set the number of columns
    # Infer number of columns from unique x values (quantized to some threshold)


    bit_count = len(bits)

    # Estimate cols as sqrt of bit count for roughly square grid
    cols = int(math.sqrt(bit_count))
    rows = math.ceil(bit_count / cols)

    print(f"🧱 Adjusted Grid: {rows} rows × {cols} cols")

    # Infer number of columns from unique x values (quantized to some threshold)
    x_positions = sorted(set(b['x'] // 10 for b in bits))  # cluster by ~10px width
    cols = len(x_positions)

    # Same for rows
    y_positions = sorted(set(b['y'] // 10 for b in bits))
    rows = len(y_positions)

    trimmed_bits, rows, cols = reconstruct_grid(binary_str, rows, cols)

    print(f"🧱 Grid: {rows} rows × {cols} cols")

    print_ascii_grid(trimmed_bits, rows, cols)

    # Visual output (black & white)
    draw_bits_image(
        bits=trimmed_bits,
        rows=rows,
        cols=cols,
        output_path="output/translated_grid.png",
        block_size=200  # or try 20 for larger blocks
    )

    decoded_text = binary_to_text(binary_str)
    if not decoded_text or not decoded_text.isprintable():
        print("⚠️ Binary decoding failed or non-printable. Using fallback text.")
        decoded_text = fallback_text

    shape_output = recreate_shape_from_bits_grid(bits, decoded_text or fallback_text)
    lines = group_bits_by_lines(bits)
    render_text_image(lines, bits, decoded_text, output_file="output/translated_preview.png")

    export_svg(bits, decoded_text, color_img)


    print("\n🔠 Decoded Text:\n", decoded_text)
    print("\n🎩 Reconstructed Shape:\n", shape_output)
    return decoded_text, shape_output

if __name__ == '__main__':
    image_path = 'test/FlagColour.1.jpeg'
    process_image_to_shape(image_path)
