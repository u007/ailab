import os
import shutil
import platform
import time

# --- Choose your ML Framework ---
# Set this to 'tensorflow', 'pytorch', or 'none'
# This determines which library's GPU check is performed.
FRAMEWORK_CHOICE = 'tensorflow' # Or 'pytorch'

# --- File Organization Setup (CPU Bound) ---
source_dir = '/path/to/your/unsorted_thousands_of_images'
base_dest_dir = '/path/to/organized_dataset'
train_dir = os.path.join(base_dest_dir, 'train')
val_dir = os.path.join(base_dest_dir, 'validation')
categories = ['cat', 'dog', 'bird'] # Example categories

# --- GPU Detection and Setup ---
device_name = 'cpu' # Default to CPU

if FRAMEWORK_CHOICE == 'tensorflow':
    try:
        import tensorflow as tf
        print(f"TensorFlow Version: {tf.__version__}")

        # List physical GPUs
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Attempt to allocate memory to verify - might be needed on some setups
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Detected TensorFlow GPU(s): {len(gpus)}")

                # Simple check for backend type (heuristic)
                gpu_name = gpus[0].name.lower() # Check name of the first GPU
                if '/physical_device:GPU:0' in gpu_name: # Standard naming
                     # Check platform for more specific type
                    if platform.system() == "Darwin" and platform.processor() == "arm":
                         # On Apple Silicon, TF uses the 'GPU' device via Metal plugin
                         # Note: TF doesn't explicitly list 'mps' like PyTorch might
                         print("TensorFlow is likely using Metal (MPS) on Apple Silicon.")
                         device_name = 'mps' # Use 'mps' for consistency if needed later
                    elif any(vendor in os.popen('lspci | grep VGA').read().lower() for vendor in ['nvidia']):
                         print("TensorFlow is likely using CUDA on NVIDIA GPU.")
                         device_name = 'cuda:0' # Standard TF CUDA device name
                    elif any(vendor in os.popen('lspci | grep VGA').read().lower() for vendor in ['amd', 'advanced micro devices']):
                         # ROCm detection in TF can be tricky, depends on build.
                         # TF often just lists it as 'GPU' even with ROCm.
                         print("TensorFlow is likely using ROCm on AMD GPU (verify TF build).")
                         device_name = 'gpu:0' # Generic GPU, could be ROCm
                    else:
                         print("TensorFlow detected a Generic GPU.")
                         device_name = 'gpu:0'
                else:
                     print("TensorFlow detected GPU, but naming unclear.")
                     device_name = 'gpu:0' # Assign generic GPU

            except RuntimeError as e:
                print(f"TensorFlow GPU check failed: {e}")
                print("Falling back to CPU for TensorFlow.")
                device_name = 'cpu'
        else:
             # Check specifically for Apple Silicon Metal support even if list_physical_devices('GPU') is empty initially
             if platform.system() == "Darwin" and platform.processor() == "arm":
                  # Sometimes the metal plugin needs explicit check, though list_physical_devices should work with tensorflow-macos
                  # This part might be redundant if tensorflow-macos is correctly installed.
                  print("No standard GPU detected by TF, but running on Apple Silicon (M1/M2/M3+). Metal (MPS) should be available if tensorflow-macos is installed.")
                  # We can't definitively confirm MPS here without running an op, keep default 'cpu' or assume 'mps'
                  # device_name = 'mps' # Optional: assume MPS if on ARM Mac
             else:
                  print("No TensorFlow GPU detected. Using CPU.")
                  device_name = 'cpu'

    except ImportError:
        print("TensorFlow not installed.")
        if FRAMEWORK_CHOICE == 'tensorflow': device_name = 'cpu'
    except Exception as e:
        print(f"An unexpected error occurred during TensorFlow GPU check: {e}")
        if FRAMEWORK_CHOICE == 'tensorflow': device_name = 'cpu'


elif FRAMEWORK_CHOICE == 'pytorch':
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")

        if torch.cuda.is_available():
            device_name = f"cuda:{torch.cuda.current_device()}"
            print(f"PyTorch using CUDA: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        elif torch.backends.mps.is_available():
             # Check for Metal Performance Shaders (MPS) on MacOS
             if torch.backends.mps.is_built():
                 device_name = "mps"
                 print("PyTorch using MPS on Apple Silicon (Mac).")
             else:
                 print("PyTorch MPS is not built. Using CPU.")
                 device_name = 'cpu'
        elif hasattr(torch, 'version') and hasattr(torch.version, 'hip') and torch.version.hip and torch.cuda.is_available():
             # ROCm detection in PyTorch - often uses CUDA APIs/variables but runs on ROCm backend
             # This check might need refinement depending on the specific PyTorch ROCm build.
             device_name = f"cuda:{torch.cuda.current_device()}" # PyTorch ROCm often uses 'cuda' identifier
             print(f"PyTorch using ROCm on AMD GPU (Device: {torch.cuda.get_device_name(torch.cuda.current_device())})")
             # Alternative check if 'hip' is explicitly needed: device_name = 'hip:...' but 'cuda' is common.

        else:
            print("PyTorch: No CUDA, MPS, or ROCm detected. Using CPU.")
            device_name = 'cpu'

    except ImportError:
        print("PyTorch not installed.")
        if FRAMEWORK_CHOICE == 'pytorch': device_name = 'cpu'
    except Exception as e:
        print(f"An unexpected error occurred during PyTorch GPU check: {e}")
        if FRAMEWORK_CHOICE == 'pytorch': device_name = 'cpu'

else:
    print("No ML framework selected for GPU check. Assuming CPU.")
    device_name = 'cpu'

print(f"\nSelected compute device: {device_name}\n")


# --- File Organization Logic (Still CPU Bound) ---

# Create destination directories if they don't exist
print("Creating directories...")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
for category in categories:
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(val_dir, category), exist_ok=True)

# --- Process Files ---
print(f"Processing files in {source_dir}...")
start_time = time.time()
file_count = 0
processed_count = 0

# Use os.scandir for potentially better performance with many files
with os.scandir(source_dir) as entries:
    for entry in entries:
        if entry.is_file() and entry.name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            file_count += 1
            file_path = entry.path

            # --- Determine Category (example logic) ---
            determined_category = None
            for category in categories:
                # Example: filename is 'cat_001.jpg' or 'train_dog_pic.png'
                if f"_{category}_" in entry.name.lower() or entry.name.lower().startswith(category):
                    determined_category = category
                    break

            if determined_category:
                # --- Decide Train/Validation Split (example: simple split) ---
                # You'd typically use a more robust method like random sampling
                # or splitting based on existing train/val lists.
                # Simple example: put images with even index number in train, odd in validation
                # (NOT recommended for real use, just illustrative)
                if processed_count % 5 != 0: # Roughly 80% train
                    dest_folder = os.path.join(train_dir, determined_category)
                else: # Roughly 20% validation
                    dest_folder = os.path.join(val_dir, determined_category)

                dest_path = os.path.join(dest_folder, entry.name)

                # --- Perform Action (copy) ---
                try:
                    # Using copy2 preserves more metadata than copy
                    shutil.copy2(file_path, dest_path)
                    processed_count += 1
                    if processed_count % 100 == 0: # Print progress update
                         print(f"  Processed {processed_count}/{file_count} files...")
                except Exception as e:
                    print(f"Error copying {entry.name} to {dest_folder}: {e}")

            # else:
            #     print(f"Could not determine category for {entry.name}, skipping.") # Optional: report skipped files

end_time = time.time()
print("\nFinished processing.")
print(f"Total files found: {file_count}")
print(f"Files copied/organized: {processed_count}")
print(f"Time taken: {end_time - start_time:.2f} seconds")

# --- Placeholder for GPU-accelerated tasks ---
print("\nFile organization complete.")
print(f"You can now proceed with ML tasks using the device: {device_name}")

# Example of how you MIGHT use the device later (using PyTorch syntax here):
# if FRAMEWORK_CHOICE == 'pytorch' and device_name != 'cpu':
#     try:
#         import torch
#         # Load your organized dataset using torch.utils.data.Dataset and DataLoader
#         # Define your model
#         # model.to(device_name) # Move model to GPU
#         # During training loop:
#         #   inputs, labels = inputs.to(device_name), labels.to(device_name) # Move data batch to GPU
#         #   outputs = model(inputs)
#         #   ... rest of training steps ...
#         print(f"PyTorch model and data would be moved to '{device_name}' for training/inference.")
#     except ImportError:
#         pass # PyTorch not installed
#
# # Similar logic would apply for TensorFlow using tf.device context manager or placing ops on GPU
# elif FRAMEWORK_CHOICE == 'tensorflow' and device_name != 'cpu':
#      try:
#          import tensorflow as tf
#          # Use tf.data to load your organized dataset
#          # Define your Keras model
#          # TensorFlow often automatically places operations on the detected GPU
#          # Or use tf.device('/gpu:0'): context manager for specific ops
#          print(f"TensorFlow operations would run on '{device_name}' (or best available GPU).")
#      except ImportError:
#          pass # TensorFlow not installed