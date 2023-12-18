from PyPDF2 import PdfReader
import fitz  # PyMuPDF
import io
from PIL import Image

def extract_images_from_pdf(pdf_path):
    # Open the PDF file
    doc = fitz.open(pdf_path)

    # List to store images
    images = []

    # Extract images
    for page_number in range(len(doc)):
        for img in doc.get_page_images(page_number):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            images.append(image)

    return images

# Example usage
pdf_path = 'input.pdf'  # Replace with your PDF file path
extracted_images = extract_images_from_pdf(pdf_path)

# Saving the extracted images
for i, image in enumerate(extracted_images):
    image_path = f"extracted_image_{i}.png"
    image.save(image_path)

len(extracted_images)  # Returns the number of images extracted

