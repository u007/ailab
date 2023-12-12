import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
from fpdf import FPDF

def find_rotation_angle(image):
    # Convert PIL image to OpenCV format
    open_cv_image = np.array(image) 
    open_cv_image = open_cv_image[:, :, ::-1].copy() 

    # Convert to grayscale
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

    # Detect edges and lines
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

    # Calculate angles
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
        angles.append(angle)

    # Return the median angle
    return np.median(angles)

def auto_rotate_image(image):
    angle = find_rotation_angle(image)
    return image.rotate(-angle, expand=True)

def pdf_to_png_auto_rotate_and_combine(pdf_path, output_pdf_path):
    # Convert PDF to images
    print('Converting PDF to images %s...', pdf_path)
    images = convert_from_path(pdf_path)
    print('auto rotating %d images', len(images))
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

# Usage
pdf_to_png_auto_rotate_and_combine('GroupHospitalisationSurgical_2309.pdf', 'output.pdf')
