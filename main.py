import sys
from config import CONFIG
from posterlike.posterlike import process_posterlike
from basic.basic import process_basic
from utils import process_image_to_shape
import pytesseract
# Path to Tesseract (adjust if needed)
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

if __name__ == '__main__':
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        expected_chars = None  # You can set this if you know the length
    else:
        image_path = "test/testColorSpaces.jpeg"  # Default test image
        expected_chars = None  # For "ThisIsATest"
    print(f"Processing image: {image_path}")
    decoded_text, shape_output, features = process_image_to_shape(image_path, expected_chars=expected_chars)
    if not isinstance(decoded_text, str):
        decoded_text = ""
    # --- Mode selection logic ---
    decoded_text_clean = decoded_text.strip()
    if 'POSTER'in decoded_text_clean :
        print("[MODE] Posterlike mode selected via marker.")
        process_posterlike(image_path, CONFIG, decoded_text, features)
    elif 'BASIC'in decoded_text_clean :
        print("[MODE] Basic mode selected via marker.")
        process_basic(image_path, CONFIG, decoded_text, features)
    elif ' ' in decoded_text_clean:
        print("[MODE] Basic mode selected via presence of spaces in decoded text.")
        process_basic(image_path, CONFIG, decoded_text, features)
    else:
        print("[MODE] Posterlike mode selected via absence of spaces in decoded text.")
        process_posterlike(image_path, CONFIG, decoded_text, features)
