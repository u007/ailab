# Excel Image Extractor

This tool extracts images from Excel files (.xlsx and .xls formats) and saves them to a specified output folder.

## Features

- Supports both modern Excel (.xlsx) and legacy Excel (.xls) formats
- Extracts images using multiple methods to handle different Excel versions
- Handles various ways Excel can store images (drawings, shapes, charts, embedded images)
- Supports both embedded and externally linked images
- Automatically detects file format and uses appropriate extraction method
- Preserves image quality and format

## Requirements

- Python 3.6+
- openpyxl (for .xlsx files)
- xlrd==1.2.0 (for .xls files, optional)
- requests (for downloading linked images)

## Installation

```bash
pip install openpyxl
pip install xlrd==1.2.0  # For .xls support
pip install requests  # For linked images support
```

## Usage

extract images from exported excel as output.html

```bash
python extract_sheet_images.py
```

download images from exported excel as output.html
```bash
python download_images.py
```

manual training
```bash
python train.py
```

The extracted images will be saved to the `output_images` folder in the current directory.

## How it works

The script uses different methods to extract images from Excel files:

1. For .xlsx files (using openpyxl):
   - Extracts images from drawings, shapes, charts, and embedded objects
   - Handles various Excel versions and storage formats

2. For .xls files (using xlrd):
   - Extracts images from the binary format
   - Requires xlrd library with formatting_info=True

## Troubleshooting

If no images are extracted:

1. Make sure the Excel file actually contains images
2. Try saving the Excel file in a different format (.xlsx or .xls)
3. For .xls files, make sure xlrd is installed with version 1.2.0
4. For linked images, ensure you have internet connectivity and the image URLs are accessible