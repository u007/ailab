import fitz  # PyMuPDF
import io
import pytesseract
from PIL import Image

def extract_text_ocr_from_page(pdf_path, page_number=0):
    with fitz.open(pdf_path) as pdf:
        page = pdf[page_number]
        text = page.get_text()
        
        # If text extraction yields results, return it.
        if text.strip():  # strip() is used to ensure we don't consider whitespace as content
            return text
        
        # If no text, attempt OCR on the page's images
        image_list = page.get_images(full=True)
        
        # In case the page has no images or text content is not found, return an empty string
        if not image_list:
            print("No images found on page", page_number)
            return ""
        
        # Assuming that the blood test results may be present as images, we use OCR to extract them
        for img_index, img in enumerate(image_list, start=1):
            xref = img[0]
            base_image = pdf.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Use BytesIO for creating an image object from the byte array returned by extract_image
            image = Image.open(io.BytesIO(image_bytes))
            text = pytesseract.image_to_string(image)
            return text  # Assume only one image contains the blood test result, return after first

    return ""  # Return an empty string if no text was extracted

def extract_text_from_first_page(pdf_path):
    # Open the provided PDF file
    with fitz.open(pdf_path) as pdf:
        # Extract text from the first page
        page = pdf[0]  # assuming the results are on the first page
        text = page.get_text()
        return text

def extract_blood_test_results(text):
    # Process the text here to extract blood test results
    # This is a placeholder function that you'll need to implement based on your PDF structure
    # For instance, you might look for certain keywords or value patterns
    # Return the extracted information
    return text  # placeholder return

# Use the function and print the results
pdf_path = '2009-11-07_LabReport_BPHealth-1.pdf'
text = extract_text_ocr_from_page(pdf_path)
print("extracted", text)
results = extract_blood_test_results(text)
print(results)
