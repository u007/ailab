import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
from fpdf import FPDF
import pytesseract
import re
import os

def find_rotation_angle(image):
    # Convert PIL image to OpenCV format
    open_cv_image = np.array(image.convert('RGB')) 
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines in the image
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)

    if lines is None:
        print('No lines were detected.')
        return 0

    # Calculate the angles of the lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
        angles.append(angle)

    # Compute the median angle
    median_angle = np.median(angles)
    return median_angle

# def find_rotation_angle(image):
#     osd = pytesseract.image_to_osd(image)
#     angle = int(re.search('(?<=Rotate: )\d+', osd).group(0))

#     # Calculate the rotation angle as a negative value
#     rotation_angle = -angle % 360
#     return rotation_angle

def auto_rotate_image(image):
    angle = find_rotation_angle(image)
    rotated = image.rotate(-angle, expand=True, fillcolor='white')

    # Create a new white background image
    white_bg = Image.new('RGB', rotated.size, (255, 255, 255))
    # Calculate the position to paste the rotated image on the white background
    position = ((white_bg.width - rotated.width) // 2, (white_bg.height - rotated.height) // 2)
    white_bg.paste(rotated, position)

    return white_bg

# def correct_orientation(image):
#     # Use Tesseract to detect orientation
#     osd = pytesseract.image_to_osd(image)
#     angle = int(re.search('(?<=Rotate: )\d+', osd).group(0))

#     # Calculate the angle to rotate
#     rotation_angle = -angle % 360

#     # Rotate the image
#     rotated_image = image.rotate(rotation_angle, expand=True, fillcolor='white')

#     return rotated_image


def pdf_to_png_auto_rotate_and_combine(pdf_path, output_pdf_path):
    # Convert PDF to images
    print('Converting PDF to images %s' % pdf_path)
    images = convert_from_path(pdf_path)
    print('auto rotating %d images' % len(images))
    # Auto-rotate and save each image
    rotated_images = [auto_rotate_image(image) for image in images]

    # Create a PDF to save the images
    pdf = FPDF(unit="pt", format=[images[0].width, images[0].height])
    
    for i, img in enumerate(rotated_images):
        pdf.add_page()
        img_path = f'temp_img_{i}.png'  # Append index to the file name
        img.save(img_path)
        pdf.image(img_path, 0, 0)

    # Save the combined PDF
    pdf.output(output_pdf_path)

def combine_images_as_pdf(images, output_pdf_path):
    # Create a PDF to save the images
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)
    pdf = FPDF(unit="pt", format=[max_width, max_height])
    
    for i, img in enumerate(images):
        pdf.add_page()
        img_path = f'temp_img_{i}.png'  # Append index to the file name
        img.save(img_path)
        pdf.image(img_path, 0, 0)

    # Save the combined PDF
    pdf.output(output_pdf_path)

def convert_pngs_in_folder_to_pdf(folder_path, output_pdf_path):
    # Get all image files sorted by name
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])

    # Load and sort images
    images = [Image.open(os.path.join(folder_path, file)) for file in image_files]

    combine_images_as_pdf(images, output_pdf_path)

# Usage
pdf_to_png_auto_rotate_and_combine('james-claim.pdf', 'output.pdf')

# convert_pngs_in_folder_to_pdf('backup', 'output.pdf')
