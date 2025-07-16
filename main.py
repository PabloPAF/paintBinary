import sys
from config import CONFIG
from posterlike.posterlike import process_posterlike
from basic.basic import process_basic
from utils import binary_to_text

if __name__ == '__main__':
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "test/flagColor.jpeg"  # Default test image
    # For demonstration, you may want to implement your own bit extraction pipeline here
    # For now, use a dummy binary string or implement your extraction logic
    binary_str = "0100100001100101011011000110110001101111"  # 'Hello' in binary
    decoded_text = binary_to_text(binary_str)
    if not isinstance(decoded_text, str):
        decoded_text = ""
    # --- Mode selection logic ---
    if decoded_text.strip().startswith('#POSTER'):
        print("[MODE] Posterlike mode selected via marker.")
        process_posterlike(image_path, CONFIG, decoded_text)
    elif decoded_text.strip().startswith('#BASIC'):
        print("[MODE] Basic mode selected via marker.")
        process_basic(image_path, CONFIG, decoded_text)
    else:
        print("[MODE] Defaulting to Posterlike mode.")
        process_posterlike(image_path, CONFIG, decoded_text)
