#!/usr/bin/env python3
"""
Dataset downloader with resume capability for the LLM training project.
This script can download various datasets with automatic resume functionality.
"""

import os
import argparse
from download_utils import ResumableDownloader, progress_bar
import zipfile
import tarfile
from pathlib import Path

# Dataset configurations
DATASET_CONFIGS = {
    'outdoor_advertising': {
        'description': 'Outdoor advertising dataset with billboards and signage',
        'files': [
            # Add actual dataset URLs here when available
            # ('https://example.com/outdoor_ads.zip', 'outdoor_ads.zip'),
        ],
        'extract_to': 'billboard'
    },
    'gantry_samples': {
        'description': 'Gantry and overhead signage samples',
        'files': [
            # Add actual dataset URLs here when available
            # ('https://example.com/gantry_samples.zip', 'gantry_samples.zip'),
        ],
        'extract_to': 'gantry-samples'
    },
    'custom': {
        'description': 'Custom dataset from provided URLs',
        'files': [],  # Will be populated from command line
        'extract_to': 'custom'
    }
}

def extract_archive(archive_path: str, extract_to: str) -> bool:
    """
    Extract archive file to specified directory.
    
    Args:
        archive_path: Path to the archive file
        extract_to: Directory to extract to
        
    Returns:
        True if extraction successful, False otherwise
    """
    archive_path = Path(archive_path)
    extract_to = Path(extract_to)
    
    if not archive_path.exists():
        print(f"Archive file not found: {archive_path}")
        return False
    
    extract_to.mkdir(parents=True, exist_ok=True)
    
    try:
        if archive_path.suffix.lower() == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                print(f"Extracting {archive_path} to {extract_to}...")
                zip_ref.extractall(extract_to)
        elif archive_path.suffix.lower() in ['.tar', '.tar.gz', '.tgz']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                print(f"Extracting {archive_path} to {extract_to}...")
                tar_ref.extractall(extract_to)
        else:
            print(f"Unsupported archive format: {archive_path.suffix}")
            return False
        
        print(f"Extraction completed: {extract_to}")
        return True
        
    except Exception as e:
        print(f"Error extracting {archive_path}: {e}")
        return False

def download_dataset(dataset_name: str, data_dir: str, custom_urls: list = None, extract: bool = True) -> bool:
    """
    Download a dataset with resume capability.
    
    Args:
        dataset_name: Name of the dataset to download
        data_dir: Directory to save the dataset
        custom_urls: List of custom URLs for 'custom' dataset
        extract: Whether to extract archives after download
        
    Returns:
        True if download successful, False otherwise
    """
    if dataset_name not in DATASET_CONFIGS:
        print(f"Unknown dataset: {dataset_name}")
        print(f"Available datasets: {list(DATASET_CONFIGS.keys())}")
        return False
    
    config = DATASET_CONFIGS[dataset_name]
    
    # Handle custom dataset URLs
    if dataset_name == 'custom' and custom_urls:
        config['files'] = [(url, Path(url).name) for url in custom_urls]
    
    if not config['files']:
        print(f"No download URLs configured for dataset: {dataset_name}")
        print("For custom datasets, provide URLs with --urls option")
        return False
    
    print(f"Downloading dataset: {dataset_name}")
    print(f"Description: {config['description']}")
    
    downloader = ResumableDownloader()
    
    def progress_callback(downloaded, total):
        progress_bar(downloaded, total)
    
    # Create data directory
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Download files
    downloaded_files = []
    for url, filename in config['files']:
        filepath = data_path / filename
        print(f"\nDownloading {filename} from {url}...")
        
        success = downloader.download_file(
            url=url,
            filepath=str(filepath),
            progress_callback=progress_callback
        )
        
        if not success:
            print(f"\nFailed to download {filename}")
            return False
        
        print(f"\nCompleted: {filename}")
        downloaded_files.append(filepath)
    
    # Extract archives if requested
    if extract:
        extract_dir = data_path / config['extract_to']
        for filepath in downloaded_files:
            if filepath.suffix.lower() in ['.zip', '.tar', '.tar.gz', '.tgz']:
                if not extract_archive(str(filepath), str(extract_dir)):
                    print(f"Warning: Failed to extract {filepath}")
    
    print(f"\nDataset download completed: {dataset_name}")
    return True

def list_datasets():
    """List available datasets."""
    print("Available datasets:")
    print("=" * 50)
    for name, config in DATASET_CONFIGS.items():
        print(f"\n{name}:")
        print(f"  Description: {config['description']}")
        print(f"  Files: {len(config['files'])} configured")
        if config['files']:
            for url, filename in config['files'][:3]:  # Show first 3 files
                print(f"    - {filename}")
            if len(config['files']) > 3:
                print(f"    ... and {len(config['files']) - 3} more")

def main():
    parser = argparse.ArgumentParser(
        description="Download datasets with resume capability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available datasets
  python download_datasets.py --list
  
  # Download a specific dataset
  python download_datasets.py --dataset outdoor_advertising --data-dir ./data
  
  # Download custom dataset from URLs
  python download_datasets.py --dataset custom --data-dir ./data \
    --urls https://example.com/file1.zip https://example.com/file2.tar.gz
  
  # Download without extracting
  python download_datasets.py --dataset outdoor_advertising --data-dir ./data --no-extract
"""
    )
    
    parser.add_argument('--dataset', '-d', type=str,
                       help='Dataset name to download')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory to save datasets (default: ./data)')
    parser.add_argument('--urls', nargs='+', type=str,
                       help='Custom URLs for custom dataset')
    parser.add_argument('--no-extract', action='store_true',
                       help='Do not extract downloaded archives')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List available datasets')
    
    args = parser.parse_args()
    
    if args.list:
        list_datasets()
        return
    
    if not args.dataset:
        print("Error: Please specify a dataset to download or use --list to see available datasets")
        parser.print_help()
        return
    
    success = download_dataset(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        custom_urls=args.urls,
        extract=not args.no_extract
    )
    
    if success:
        print("\n✅ Download completed successfully!")
    else:
        print("\n❌ Download failed!")
        exit(1)

if __name__ == '__main__':
    main()