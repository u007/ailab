# Download Guide: Resumable Downloads

This guide explains how to use the resumable download functionality in the LLM training project.

## Overview

The project now includes robust resumable download capabilities for:
- **Model downloads**: Automatic resume for HuggingFace models
- **Dataset downloads**: Custom resumable downloader for any file
- **Progress tracking**: Real-time progress bars and status updates
- **Integrity verification**: Hash and size verification for downloaded files

## Features

### ✅ Resumable Downloads
- Automatically resumes interrupted downloads
- Supports HTTP range requests
- Handles network timeouts and retries
- Verifies file integrity after download

### ✅ Progress Tracking
- Real-time progress bars
- Download speed and ETA estimates
- Human-readable file sizes
- Status updates for each file

### ✅ Error Handling
- Automatic retries with exponential backoff
- Graceful fallback for servers without range support
- Detailed error messages and logging
- Corruption detection and re-download

## Usage

### 1. Model Downloads (Automatic)

When you run the training script, models are automatically downloaded with resume capability:

```bash
python train.py
```

The script will:
- Download the Qwen2-VL model to a local cache
- Resume any interrupted downloads automatically
- Verify model integrity
- Use cached models for subsequent runs

### 2. Dataset Downloads (Manual)

Use the dedicated dataset downloader for custom datasets:

```bash
# List available datasets
python download_datasets.py --list

# Download a specific dataset
python download_datasets.py --dataset outdoor_advertising --data-dir ./data

# Download custom files from URLs
python download_datasets.py --dataset custom --data-dir ./data \
  --urls https://example.com/dataset1.zip https://example.com/dataset2.tar.gz

# Download without extracting archives
python download_datasets.py --dataset outdoor_advertising --data-dir ./data --no-extract
```

### 3. Programmatic Usage

Use the `ResumableDownloader` class directly in your code:

```python
from download_utils import ResumableDownloader, progress_bar

# Create downloader instance
downloader = ResumableDownloader(
    chunk_size=8192,      # Download chunk size
    max_retries=3,        # Maximum retry attempts
    retry_delay=1.0       # Delay between retries
)

# Progress callback function
def progress_callback(downloaded, total):
    progress_bar(downloaded, total)

# Download a file
success = downloader.download_file(
    url="https://example.com/large-file.zip",
    filepath="./downloads/large-file.zip",
    expected_size=1024*1024*100,  # Optional: expected size in bytes
    expected_hash="sha256_hash",   # Optional: expected SHA256 hash
    progress_callback=progress_callback
)

if success:
    print("Download completed successfully!")
else:
    print("Download failed!")
```

## Configuration

### Download Settings

You can customize download behavior by modifying the `ResumableDownloader` parameters:

```python
downloader = ResumableDownloader(
    chunk_size=16384,     # Larger chunks for faster downloads
    max_retries=5,        # More retries for unreliable connections
    retry_delay=2.0       # Longer delay between retries
)
```

### Cache Directories

The training script uses these cache directories:
- `./models_cache/`: HuggingFace model cache
- `./data/`: Dataset storage
- `./results/`: Training outputs

## File Verification

### Automatic Verification

Downloaded files are automatically verified using:
- **Size verification**: Compares actual vs expected file size
- **Hash verification**: Validates file integrity using SHA256/MD5
- **Corruption detection**: Re-downloads corrupted files

### Manual Verification

```python
from download_utils import ResumableDownloader

downloader = ResumableDownloader()

# Verify a downloaded file
is_valid = downloader._verify_file(
    filepath="./downloads/file.zip",
    expected_size=1024*1024,
    expected_hash="abc123...",
    hash_algorithm="sha256"
)
```

## Troubleshooting

### Common Issues

1. **Download Stuck at 0%**
   - Check internet connection
   - Verify URL is accessible
   - Check if server supports range requests

2. **Hash Verification Failed**
   - File may be corrupted during download
   - Server may have updated the file
   - Delete the file and re-download

3. **Permission Errors**
   - Ensure write permissions to download directory
   - Check available disk space

4. **Memory Issues**
   - Reduce `chunk_size` for large files
   - Close other applications to free memory

### Debug Mode

Enable verbose logging for debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your download code here
```

### Manual Resume

If a download fails, simply run the same command again. The downloader will:
1. Check if the file already exists
2. Verify its integrity
3. Resume from where it left off if incomplete
4. Skip download if file is already complete and valid

## Performance Tips

### Optimize Download Speed

1. **Increase chunk size** for faster downloads:
   ```python
   downloader = ResumableDownloader(chunk_size=32768)
   ```

2. **Use multiple connections** for large files (manual implementation needed)

3. **Choose closer mirrors** when available

### Reduce Memory Usage

1. **Decrease chunk size** for memory-constrained environments:
   ```python
   downloader = ResumableDownloader(chunk_size=4096)
   ```

2. **Disable progress callbacks** for headless environments:
   ```python
   downloader.download_file(url, filepath)  # No progress_callback
   ```

## Examples

### Example 1: Download Training Data

```bash
# Download outdoor advertising dataset
python download_datasets.py --dataset outdoor_advertising --data-dir ./data

# Train the model
python train.py
```

### Example 2: Resume Interrupted Download

```bash
# Start download (gets interrupted)
python download_datasets.py --dataset custom --urls https://example.com/large-dataset.zip

# Resume download (automatically continues from where it left off)
python download_datasets.py --dataset custom --urls https://example.com/large-dataset.zip
```

### Example 3: Batch Download Multiple Files

```bash
# Download multiple datasets
python download_datasets.py --dataset custom --data-dir ./data \
  --urls \
    https://example.com/dataset1.zip \
    https://example.com/dataset2.tar.gz \
    https://example.com/dataset3.zip
```

## Integration with Training

The resumable download functionality is seamlessly integrated into the training pipeline:

1. **Model Download**: Automatically downloads and caches the Qwen2-VL model
2. **Resume Support**: Resumes interrupted model downloads
3. **Cache Management**: Reuses downloaded models across training runs
4. **Error Handling**: Gracefully handles download failures with fallbacks

This ensures that your training can continue even after network interruptions or system restarts.