from bs4 import BeautifulSoup
import json
from pathlib import Path

def extract_images_by_section():
    # Read the HTML file
    html_path = Path('output.html')
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all sections (they start with 'Sheet')
    sections = {}
    current_section = None
    
    # Process all elements in order
    for element in soup.find_all(['h1', 'h2', 'img']):
        if element.name in ['h1', 'h2'] and 'Sheet' in element.text:
            current_section = element.text.strip()
            sections[current_section] = []
        elif element.name == 'img' and current_section:
            parent_link = element.find_parent('a')
            if parent_link and parent_link.get('href'):
                sections[current_section].append(parent_link.get('href'))

    # Save results to a JSON file
    output_file = 'image_urls_by_section.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sections, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\nExtracted images by section:")
    for section, images in sections.items():
        print(f"\n{section}:")
        print(f"Number of images: {len(images)}")

if __name__ == '__main__':
    extract_images_by_section()