# PaintBinary

A Python-based image processing tool that extracts colored regions from images and renders text densely within those regions in a collage-like style. The tool can decode binary data from images and create artistic text-based representations.
This version only works with binary code.

## TODO
Implement other decoding methods: hex, B64
improve  character recognition
    - Implement first AI for 0 and 1 handwriting
    - Maybe forthe others in the future
Improve decoding: check if word extracted makes sense horizontally/ vertically, choose the better one.
Recognition of which encoding is being seen
Mixed encoding


## Features

- **Binary Data Extraction**: Uses OCR to detect and extract binary data (0s and 1s) from images
- **Color Region Detection**: Identifies colored macroblocks in images using contour detection
- **Text Rendering**: Fills detected regions with decoded text using various rendering styles:
  - Grid-based text filling
  - Collage-style overlapping characters
  - Adaptive font sizing based on region size
- **Multiple Output Formats**: Generates PNG images, SVG files, and text files
- **Macroblock Merging**: Intelligently merges nearby colored regions to reduce fragmentation
- **Debug Visualizations**: Creates debug images showing detected regions and processing steps

## Installation

### Prerequisites

- Python 3.7+
- OpenCV (`cv2`)
- PIL/Pillow
- pytesseract
- numpy
- svgwrite

### Setup

1. **Install Tesseract OCR** (required for binary data extraction):
   ```bash
   # macOS (using Homebrew)
   brew install tesseract
   
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr
   ```

2. **Install Python dependencies**:
   ```bash
   pip install opencv-python pillow pytesseract numpy svgwrite
   ```

3. **Configure Tesseract path** (if needed):
   - The default path is set for macOS: `/opt/homebrew/bin/tesseract`
   - Update the path in `main.py` if your installation is different

## Usage

### Basic Usage

```python
from main import process_image_to_shape

# Process an image
decoded_text, shape_output = process_image_to_shape("path/to/your/image.jpeg")
```

### Command Line

```bash
python main.py
```

The script will process the image specified in `main.py` (currently set to `"test/flagColor.jpeg"`).

## Project Structure

```
paintBinary/
├── main.py                 # Main processing script
├── output/                 # Generated output files
│   ├── debug_colored_contours.png
│   ├── debug_thresh_check.png
│   ├── debug_thresh.png
│   ├── translated_blocks.png
│   ├── translated_preview.png
│   ├── translated.svg
│   └── translated.txt
├── test/                   # Test images
│   ├── flagColor.jpeg
│   └── ...
├── FlagOrigin/            # Original flag images
└── raw_binary.bin        # Raw binary data
```

## How It Works

### 1. Image Preprocessing
- Loads and resizes the input image
- Applies contrast enhancement using CLAHE
- Performs adaptive thresholding to create binary image
- Applies morphological operations for cleanup

### 2. Binary Data Extraction
- Uses Tesseract OCR to detect '0' and '1' characters
- Extracts position and color information for each detected bit
- Converts binary string to ASCII text

### 3. Color Region Detection
- Detects colored contours in the image
- Groups nearby regions by color similarity and proximity
- Merges macroblocks to reduce fragmentation

### 4. Text Rendering
- Fills detected regions with decoded text
- Uses adaptive font sizing (11pt to 250pt)
- Supports both grid-style and collage-style rendering
- Applies region colors to text

## Output Files

- **`translated_preview.png`**: Preview with text rendered at bit positions
- **`translated_blocks.png`**: Text rendered in detected color blocks
- **`translated_macroblocks.png`**: Collage-style rendering in macroblocks
- **`debug_colorBlocks.png`**: Debug view showing macroblock boundaries
- **`translated.svg`**: Vector output with text positioned at bit locations
- **`translated.txt`**: Raw decoded text

## Configuration

### Key Parameters

- **Font Size Range**: 11pt to 250pt (adjustable in code)
- **Macroblock Merging**: Proximity threshold (200px), color distance (10)
- **Minimum Area**: 50 pixels for macroblock detection
- **Font Path**: `/Library/Fonts/Courier New.ttf` (macOS)

### Customization

You can modify these parameters in `main.py`:

```python
# Font size range
min_font_pt = 11
max_font_pt = 250

# Macroblock merging parameters
merged_macroblocks = merge_macroblocks(macroblocks, 
                                     proximity_thresh=120, 
                                     color_thresh=20)

# Minimum area for detection
min_area = 50
```

## Troubleshooting

### Common Issues

1. **Tesseract not found**: Update the path in `main.py`
2. **Font not found**: Install system fonts or update font paths
3. **No regions detected**: Adjust color thresholds in contour detection
4. **Large blank spaces**: Increase minimum font size or adjust macroblock merging

### Debug Outputs

The tool generates several debug images:
- `debug_thresh.png`: Binary threshold image
- `debug_colored_contours.png`: Detected color regions
- `debug_macroblocks.png`: Macroblock boundaries

## Technical Details

### Algorithms Used

- **OCR**: Tesseract for binary character recognition
- **Contour Detection**: OpenCV for color region identification
- **Color Clustering**: Distance-based grouping of similar colors
- **Font Rendering**: PIL/Pillow for text rendering with rotation

### Performance Considerations

- Large images are automatically downscaled to 4096px max dimension
- Progressive font size testing optimizes rendering speed
- Macroblock merging reduces processing overhead

## License

This project is for educational and artistic purposes. Please ensure you have rights to any images you process.

## Contributing

Feel free to submit issues and enhancement requests. The code is structured for easy modification of rendering styles and detection parameters. 