from bs4 import BeautifulSoup
import json
from pathlib import Path
import os

def extract_sheet_images():
    input_fld = Path('input.fld')
    
    # Process each sheet HTML file
    for i in range(1, 22):  # Based on the 21 sheets seen in input.htm
        sheet_num = str(i).zfill(3)
        sheet_file = input_fld / f'sheet{sheet_num}.htm'
        
        if not sheet_file.exists():
            print(f"Warning: {sheet_file} not found")
            continue
            
        # Read and parse the sheet HTML
        with open(sheet_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find all <a> tags containing images
        image_urls = []
        for a_tag in soup.find_all('a'):
            if a_tag.find('img'):
                href = a_tag.get('href')
                if href:
                    image_urls.append(href)
        
        # Save to JSON file
        output_file = f'sheet{sheet_num}.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'sheet_number': sheet_num,
                'image_urls': image_urls
            }, f, indent=2, ensure_ascii=False)
        
        print(f"Processed sheet{sheet_num}.htm - Found {len(image_urls)} images")

if __name__ == '__main__':
    extract_sheet_images()