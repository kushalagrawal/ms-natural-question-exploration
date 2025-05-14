import os
import gzip
import shutil # For efficient file copying
import multiprocessing
import logging
from tqdm import tqdm
import time
import sys # For exit

# --- Configuration ---
# Base directory containing the subdirectories (like 'v1.0')
# If 'v1.0' is in the current directory, set this to '.'
BASE_SOURCE_DIR = '.' # Adjust if 'v1.0' is elsewhere

# Subdirectories within BASE_SOURCE_DIR to scan for archives
# Relative paths from BASE_SOURCE_DIR
SUBDIRS_TO_SCAN = ['v1.0/dev', 'v1.0/train', 'v1.0/sample']

# Base directory where the decompressed files will be placed
TARGET_BASE_DIR = 'unarchived'

# Number of parallel processes (adjust based on your CPU cores)
# None uses os.cpu_count(), or set a specific number like 4 or 8.
NUM_PROCESSES = None # Use None for automatic detection

# --- Setup ---
# Ensure base target directory exists
os.makedirs(TARGET_BASE_DIR, exist_ok=True)

# Setup logging
log_file_path = os.path.join(TARGET_BASE_DIR, "decompression.log") # Log file name changed
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler() # Also print logs to console
    ]
)

# --- Decompression Function (for a single file) ---
def decompress_single_gz(source_gz_path, target_jsonl_path):
    """
    Decompresses a single .gz file to a target path.

    Args:
        source_gz_path (str): Path to the source .jsonl.gz file.
        target_jsonl_path (str): Path where the decompressed .jsonl file should be saved.
    """
    source_filename = os.path.basename(source_gz_path)
    target_filename = os.path.basename(target_jsonl_path)
    # Get relative path for logging clarity
    relative_target_dir = os.path.relpath(os.path.dirname(target_jsonl_path), TARGET_BASE_DIR)

    logging.info(f"Starting decompression: {source_filename} -> {os.path.join(relative_target_dir, target_filename)}")
    start_time = time.time()
    try:
        # Ensure the target directory exists
        os.makedirs(os.path.dirname(target_jsonl_path), exist_ok=True)

        # Open source (.gz) and target (.jsonl) files
        with gzip.open(source_gz_path, 'rb') as f_in:
            with open(target_jsonl_path, 'wb') as f_out:
                # Copy the decompressed content efficiently
                shutil.copyfileobj(f_in, f_out)

        end_time = time.time()
        logging.info(f"Successfully decompressed {source_filename} -> {os.path.join(relative_target_dir, target_filename)} in {end_time - start_time:.2f} seconds.")
        return True, source_filename # Indicate success
    except gzip.BadGzipFile as e:
        logging.error(f"Error reading gzip file {source_filename}: {e}. It might be corrupted or not a valid gzip file.")
        # Optionally remove potentially incomplete target file
        if os.path.exists(target_jsonl_path):
            try:
                os.remove(target_jsonl_path)
            except OSError:
                pass # Ignore removal error
        return False, source_filename # Indicate failure
    except FileNotFoundError:
        logging.error(f"Source file not found during decompression: {source_gz_path}")
        return False, source_filename # Indicate failure
    except Exception as e:
        logging.error(f"An unexpected error occurred while decompressing {source_filename} -> {target_filename}: {e}")
        # Optionally remove potentially incomplete target file
        if os.path.exists(target_jsonl_path):
             try:
                 os.remove(target_jsonl_path)
             except OSError:
                 pass # Ignore removal error
        return False, source_filename # Indicate failure

# --- Main Execution ---
if __name__ == "__main__":
    # This check is important for multiprocessing
    multiprocessing.freeze_support()

    logging.info(f"Starting Natural Questions dataset decompression (Hierarchical).")
    logging.info(f"Base Source directory: {os.path.abspath(BASE_SOURCE_DIR)}")
    logging.info(f"Scanning subdirectories: {SUBDIRS_TO_SCAN}")
    logging.info(f"Base Target directory: {os.path.abspath(TARGET_BASE_DIR)}")

    # 1. Find all .jsonl.gz files within the specified subdirectories
    tasks = [] # List of tuples: (source_gz_path, target_jsonl_path)
    files_found_count = 0
    missing_subdirs = []

    for subdir_rel_path in SUBDIRS_TO_SCAN:
        current_scan_dir = os.path.join(BASE_SOURCE_DIR, subdir_rel_path)
        logging.info(f"Scanning directory: {current_scan_dir}")

        if not os.path.isdir(current_scan_dir):
            logging.warning(f"Subdirectory not found: {current_scan_dir}. Skipping.")
            missing_subdirs.append(subdir_rel_path)
            continue

        try:
            for item in os.listdir(current_scan_dir):
                # Look specifically for .jsonl.gz files
                if item.lower().endswith(".jsonl.gz"): # Use lower() for case-insensitivity
                    source_gz_path = os.path.join(current_scan_dir, item)
                    if os.path.isfile(source_gz_path):
                        # Calculate the specific target directory
                        specific_target_dir = os.path.join(TARGET_BASE_DIR, subdir_rel_path)
                        # Calculate the target filename by removing .gz
                        target_filename = item[:-3] # Remove the last 3 characters (".gz")
                        target_jsonl_path = os.path.join(specific_target_dir, target_filename)

                        tasks.append((source_gz_path, target_jsonl_path))
                        files_found_count += 1
        except Exception as e:
            logging.error(f"Error scanning directory '{current_scan_dir}': {e}")
            # Log and continue with other directories

    if not tasks:
        logging.warning(f"No .jsonl.gz files found in the specified subdirectories: {SUBDIRS_TO_SCAN}. Nothing to decompress.")
        if missing_subdirs:
             logging.warning(f"The following subdirectories were missing: {missing_subdirs}")
        sys.exit(0)

    logging.info(f"Found {files_found_count} .jsonl.gz files across specified subdirectories.")

    # 2. Run decompression in parallel
    successful_decompressions = 0
    failed_decompressions = []
    total_start_time = time.time()

    try:
        # Create a pool of worker processes
        with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
            actual_processes = pool._processes if hasattr(pool, '_processes') else NUM_PROCESSES if NUM_PROCESSES else os.cpu_count()
            logging.info(f"Starting decompression with {actual_processes} processes.")

            # Use starmap as the function takes multiple arguments
            results_iterator = pool.starmap(decompress_single_gz, tasks)

            # Wrap with tqdm for a progress bar
            for success, filename in tqdm(results_iterator, total=len(tasks), desc="Decompressing files"):
                if success:
                    successful_decompressions += 1
                else:
                    failed_decompressions.append(filename)

    except Exception as e:
        logging.error(f"A critical error occurred during parallel processing: {e}")

    total_end_time = time.time()
    logging.info("-" * 30)
    logging.info(f"Decompression process finished in {total_end_time - total_start_time:.2f} seconds.")
    logging.info(f"Successfully decompressed: {successful_decompressions}/{len(tasks)} files.")

    if failed_decompressions:
        logging.warning(f"Failed to decompress {len(failed_decompressions)} files:")
        for fname in sorted(list(set(failed_decompressions))):
            logging.warning(f"  - {fname}")
    logging.info(f"Check '{log_file_path}' for detailed logs.")
    logging.info(f"Decompressed contents are in subdirectories within: {os.path.abspath(TARGET_BASE_DIR)}")
