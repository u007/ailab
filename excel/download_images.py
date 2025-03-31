import os
import json
import aiohttp
import asyncio
from pathlib import Path
from urllib.parse import urlparse

async def download_image(session, url, save_path):
    try:
        async with session.get(url) as response:
            if response.status == 200:
                content = await response.read()
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # Save the image
                with open(save_path, 'wb') as f:
                    f.write(content)
                print(f"Downloaded: {save_path}")
            else:
                print(f"Failed to download {url}: HTTP {response.status}")
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")

async def process_sheet(session, sheet_path):
    try:
        # Read the JSON file
        with open(sheet_path, 'r') as f:
            data = json.load(f)
        
        sheet_number = data.get('sheet_number', '')
        image_urls = data.get('image_urls', [])
        
        if not sheet_number or not image_urls:
            print(f"Skipping {sheet_path}: Missing required data")
            return
        
        # Create folder for this sheet
        folder_name = f"sheet{sheet_number}_images"
        
        # Process each URL
        tasks = []
        for i, url in enumerate(image_urls):
            # Extract file extension from URL or default to .jpg
            parsed_url = urlparse(url)
            file_name = f"image_{i+1}.jpg"
            save_path = os.path.join('output_images', folder_name, file_name)
            # Skip if file already exists
            if not os.path.exists(save_path):
                task = download_image(session, url, save_path)
                tasks.append(task)
            else:
                print(f"Skipping {save_path}: File already exists")
        
        await asyncio.gather(*tasks)
        
    except Exception as e:
        print(f"Error processing {sheet_path}: {str(e)}")

async def main():
    # Create output directory
    os.makedirs('output_images', exist_ok=True)
    
    # Get all sheet JSON files
    sheet_files = sorted(Path('.').glob('sheet*.json'))
    
    # Create HTTP session
    async with aiohttp.ClientSession() as session:
        tasks = [process_sheet(session, sheet_file) for sheet_file in sheet_files]
        await asyncio.gather(*tasks)

if __name__ == '__main__':
    asyncio.run(main())