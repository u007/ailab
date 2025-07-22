import os
import requests
import hashlib
from typing import Optional, Callable
from pathlib import Path
import time

class ResumableDownloader:
    """A utility class for resumable downloads with progress tracking."""
    
    def __init__(self, chunk_size: int = 8192, max_retries: int = 3, retry_delay: float = 1.0):
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def download_file(
        self, 
        url: str, 
        filepath: str, 
        expected_size: Optional[int] = None,
        expected_hash: Optional[str] = None,
        hash_algorithm: str = 'sha256',
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> bool:
        """
        Download a file with resume capability.
        
        Args:
            url: URL to download from
            filepath: Local path to save the file
            expected_size: Expected file size in bytes (optional)
            expected_hash: Expected hash of the complete file (optional)
            hash_algorithm: Hash algorithm to use ('sha256', 'md5', etc.)
            progress_callback: Callback function for progress updates (downloaded, total)
            
        Returns:
            True if download successful, False otherwise
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file already exists and is complete
        if filepath.exists():
            if self._verify_file(filepath, expected_size, expected_hash, hash_algorithm):
                print(f"File {filepath} already exists and is valid. Skipping download.")
                return True
            else:
                print(f"File {filepath} exists but is incomplete or corrupted. Resuming download.")
        
        # Determine resume position
        resume_pos = filepath.stat().st_size if filepath.exists() else 0
        
        for attempt in range(self.max_retries):
            try:
                # Set up headers for resume
                headers = {}
                if resume_pos > 0:
                    headers['Range'] = f'bytes={resume_pos}-'
                
                # Make request
                response = requests.get(url, headers=headers, stream=True, timeout=30)
                
                # Check if server supports range requests
                if resume_pos > 0 and response.status_code not in [206, 200]:
                    print(f"Server doesn't support range requests. Starting from beginning.")
                    resume_pos = 0
                    response = requests.get(url, stream=True, timeout=30)
                
                response.raise_for_status()
                
                # Get total size
                if 'content-length' in response.headers:
                    total_size = int(response.headers['content-length'])
                    if resume_pos > 0 and response.status_code == 206:
                        total_size += resume_pos
                elif expected_size:
                    total_size = expected_size
                else:
                    total_size = None
                
                # Open file for writing
                mode = 'ab' if resume_pos > 0 else 'wb'
                with open(filepath, mode) as f:
                    downloaded = resume_pos
                    
                    for chunk in response.iter_content(chunk_size=self.chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # Call progress callback
                            if progress_callback and total_size:
                                progress_callback(downloaded, total_size)
                
                # Verify download
                if self._verify_file(filepath, expected_size, expected_hash, hash_algorithm):
                    print(f"Download completed successfully: {filepath}")
                    return True
                else:
                    print(f"Download verification failed for {filepath}")
                    if attempt < self.max_retries - 1:
                        print(f"Retrying... (attempt {attempt + 2}/{self.max_retries})")
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        return False
                        
            except Exception as e:
                print(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    print(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                    resume_pos = filepath.stat().st_size if filepath.exists() else 0
                else:
                    print(f"All download attempts failed for {url}")
                    return False
        
        return False
    
    def _verify_file(
        self, 
        filepath: Path, 
        expected_size: Optional[int] = None,
        expected_hash: Optional[str] = None,
        hash_algorithm: str = 'sha256'
    ) -> bool:
        """
        Verify file integrity.
        
        Args:
            filepath: Path to the file to verify
            expected_size: Expected file size in bytes
            expected_hash: Expected hash of the file
            hash_algorithm: Hash algorithm to use
            
        Returns:
            True if file is valid, False otherwise
        """
        if not filepath.exists():
            return False
        
        # Check file size
        if expected_size is not None:
            actual_size = filepath.stat().st_size
            if actual_size != expected_size:
                print(f"Size mismatch: expected {expected_size}, got {actual_size}")
                return False
        
        # Check file hash
        if expected_hash is not None:
            actual_hash = self._calculate_hash(filepath, hash_algorithm)
            if actual_hash.lower() != expected_hash.lower():
                print(f"Hash mismatch: expected {expected_hash}, got {actual_hash}")
                return False
        
        return True
    
    def _calculate_hash(self, filepath: Path, algorithm: str = 'sha256') -> str:
        """
        Calculate hash of a file.
        
        Args:
            filepath: Path to the file
            algorithm: Hash algorithm to use
            
        Returns:
            Hex digest of the file hash
        """
        hash_obj = hashlib.new(algorithm)
        
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(self.chunk_size), b''):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()

def progress_bar(downloaded: int, total: int, width: int = 50):
    """
    Simple progress bar for download progress.
    
    Args:
        downloaded: Bytes downloaded so far
        total: Total bytes to download
        width: Width of the progress bar
    """
    if total <= 0:
        return
    
    percent = (downloaded / total) * 100
    filled = int(width * downloaded // total)
    bar = '█' * filled + '░' * (width - filled)
    
    # Convert bytes to human readable format
    def format_bytes(bytes_val):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_val < 1024.0:
                return f"{bytes_val:.1f}{unit}"
            bytes_val /= 1024.0
        return f"{bytes_val:.1f}TB"
    
    downloaded_str = format_bytes(downloaded)
    total_str = format_bytes(total)
    
    print(f"\r{bar} {percent:.1f}% ({downloaded_str}/{total_str})", end='', flush=True)

# Example usage
if __name__ == "__main__":
    downloader = ResumableDownloader()
    
    # Example download with progress bar
    def progress_callback(downloaded, total):
        progress_bar(downloaded, total)
    
    # Download a file
    success = downloader.download_file(
        url="https://example.com/large-file.zip",
        filepath="./downloads/large-file.zip",
        progress_callback=progress_callback
    )
    
    if success:
        print("\nDownload completed successfully!")
    else:
        print("\nDownload failed!")