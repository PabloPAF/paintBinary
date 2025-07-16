"""
Shared utility functions for paintBinary project.
"""
import re

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