import os
import json
import re
import logging
import multiprocessing
from functools import partial
import time
import sys
from collections import defaultdict
import shutil # For removing existing directories and temp dirs
import tempfile # For creating temporary directories
import uuid # For unique temporary file names
import pathlib # For potentially more robust path handling

import datasets # For saving/loading Arrow format
from tqdm import tqdm

# --- Dependencies ---
try:
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
except ImportError:
    print("Error: 'transformers' library not found. Please install it: pip install transformers torch datasets")
    sys.exit(1)

# --- Configuration ---
# <<< *** EDIT THIS: Set the ABSOLUTE path to your project directory *** >>>
# Example: BASE_DATA_DIR = 'C:/Users/YourUser/Documents/natural-questions'
# Example: BASE_DATA_DIR = '/home/youruser/projects/natural-questions'
BASE_DATA_DIR = '.' # Replace '.' with the full absolute path

# Construct source paths using the absolute base directory
SOURCE_DATA_BASE = os.path.join(BASE_DATA_DIR, 'unarchived', 'v1.0')
TRAIN_SOURCE_DIR = os.path.join(SOURCE_DATA_BASE, 'train')
DEV_SOURCE_DIR = os.path.join(SOURCE_DATA_BASE, 'dev')

# Define the FINAL output directory using the absolute base directory
FINAL_OUTPUT_DIR = os.path.join(BASE_DATA_DIR, 'nq_processed_final')

# <<< *** UPDATED: Directory for temporary files during processing *** >>>
# Create a subdirectory within the project directory for temp files
TEMP_PROCESSING_BASE_DIR = os.path.join(BASE_DATA_DIR, 'temp_preprocessing_files')

# --- Tokenizer and Model Config ---
TOKENIZER_CHECKPOINT = "bert-base-uncased"
MAX_SEQ_LENGTH = 384
DOC_STRIDE = 128

# --- Multiprocessing Config ---
NUM_PROCESSES = None # None uses os.cpu_count()

# --- Setup ---
os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True) # Ensure final output base dir exists
# <<< *** NEW: Ensure the base temporary directory exists *** >>>
os.makedirs(TEMP_PROCESSING_BASE_DIR, exist_ok=True)
# <<< *** END NEW *** >>>


# Setup logging
log_file_path = os.path.join(FINAL_OUTPUT_DIR, "preprocessing.log") # Log in final dir
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='w'), # Overwrite log each run
        logging.StreamHandler()
    ]
)
# Separate debug logger
debug_logger = logging.getLogger('nq_debug')
debug_logger.setLevel(logging.INFO) # Keep INFO level to avoid huge logs
debug_handler = logging.FileHandler(log_file_path, mode='a')
debug_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s'))
debug_logger.addHandler(debug_handler)
debug_logger.propagate = False


# --- Helper Functions ---
def clean_html_basic(html_content):
    """ Basic HTML cleaner - removes tags, collapses whitespace """
    if not html_content: return ""
    html_content = re.sub(r'</(li|td|th|p|div)>\s*<', r' \g<0>', html_content, flags=re.IGNORECASE)
    clean_text = re.sub(r'<[^>]+>', ' ', html_content)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text

# --- Preprocessing Function (Worker - Unchanged from v2) ---
def preprocess_and_save_single_file(filepath, tokenizer_name, max_seq_len, doc_stride, temp_dir):
    """
    Processes a single .jsonl file by generating features incrementally,
    saves them to a temporary Arrow file using Dataset.from_generator,
    and returns the path to the temporary file.
    """
    local_debug_logger = logging.getLogger('nq_debug')
    output_arrow_path = None
    file_processed_lines = 0
    skipped_lines = 0
    total_features_generated = 0

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    except Exception as e:
        logging.error(f"Worker failed to load tokenizer '{tokenizer_name}': {e}")
        return None

    def feature_generator():
        nonlocal skipped_lines, file_processed_lines, total_features_generated
        try:
            with open(filepath, 'rt', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    file_processed_lines += 1
                    try:
                        example = json.loads(line)
                        example_id = example.get('example_id', 'N/A')
                        raw_question = example.get('question_text')
                        raw_html = example.get('document_html')
                        raw_annotations = example.get('annotations')

                        local_debug_logger.debug(f"Line {line_num} | ID: {example_id} | Processing...")

                        question_text = raw_question.strip() if isinstance(raw_question, str) else None
                        document_html = raw_html if isinstance(raw_html, str) else None
                        annotations = raw_annotations if isinstance(raw_annotations, list) and raw_annotations else None

                        if not question_text or not document_html or not annotations:
                            logging.warning(f"Skipping line {line_num} in {filepath}: Missing essential fields.")
                            skipped_lines += 1; continue

                        annotation = annotations[0]
                        context_text = clean_html_basic(document_html)
                        if not context_text:
                             logging.warning(f"Skipping line {line_num} (ID: {example_id}): Empty context.")
                             skipped_lines += 1; continue

                        # Answer identification logic...
                        short_answers = annotation.get('short_answers', [])
                        answer_start_byte, answer_end_byte = -1, -1
                        answer_start_char, answer_end_char = -1, -1
                        answer_text = ""
                        if short_answers:
                            for ans in short_answers:
                                if ('start_byte' in ans and 'end_byte' in ans and
                                    isinstance(ans['start_byte'], int) and isinstance(ans['end_byte'], int) and
                                    ans['start_byte'] >= 0 and ans['end_byte'] >= 0 and ans['start_byte'] < ans['end_byte']):
                                    answer_start_byte, answer_end_byte = ans['start_byte'], ans['end_byte']
                                    try:
                                        ans_text_bytes = document_html.encode('utf-8')[answer_start_byte:answer_end_byte]
                                        answer_text = ans_text_bytes.decode('utf-8', errors='ignore').strip()
                                        context_char_start = context_text.find(answer_text)
                                        if context_char_start != -1 and answer_text:
                                            answer_start_char = context_char_start
                                            answer_end_char = context_char_start + len(answer_text)
                                        else: answer_start_byte, answer_end_byte, answer_text = -1, -1, ""; answer_start_char, answer_end_char = -1, -1
                                    except Exception: answer_start_byte, answer_end_byte, answer_text = -1, -1, ""; answer_start_char, answer_end_char = -1, -1
                                    if answer_start_char != -1: break

                        # Tokenization logic...
                        tokenized_inputs = tokenizer(
                            question_text, context_text, truncation="only_second",
                            max_length=max_seq_len, stride=doc_stride,
                            return_overflowing_tokens=True, return_offsets_mapping=True,
                            padding="max_length", return_token_type_ids=True
                        )
                        offset_mapping = tokenized_inputs.pop("offset_mapping")

                        # Feature creation and mapping logic...
                        for i, offsets in enumerate(offset_mapping):
                            feature = {"example_id": str(example_id),
                                       "input_ids": tokenized_inputs["input_ids"][i],
                                       "attention_mask": tokenized_inputs["attention_mask"][i],
                                       "start_positions": 0, "end_positions": 0}
                            if "token_type_ids" in tokenized_inputs:
                                 feature["token_type_ids"] = tokenized_inputs["token_type_ids"][i]
                            sequence_ids = tokenized_inputs.sequence_ids(i)
                            context_indices = [idx for idx, seq_id in enumerate(sequence_ids) if seq_id == 1]
                            if context_indices:
                                context_token_start_index = context_indices[0]
                                context_token_end_index = context_indices[-1]
                                if answer_start_char != -1 and answer_end_char != -1:
                                    feature_context_char_start = offsets[context_token_start_index][0]
                                    feature_context_char_end = offsets[context_token_end_index][1]
                                    if (answer_start_char >= feature_context_char_start and
                                        answer_end_char <= feature_context_char_end):
                                        # Mapping logic...
                                        start_token_found, end_token_found = False, False
                                        temp_start_pos, temp_end_pos = 0, 0
                                        for token_idx in range(context_token_start_index, context_token_end_index + 1):
                                            if offsets[token_idx][0] <= answer_start_char < offsets[token_idx][1] or \
                                               (offsets[token_idx][0] == answer_start_char and offsets[token_idx][1] > answer_start_char):
                                                temp_start_pos = token_idx; start_token_found = True; break
                                        for token_idx in range(context_token_end_index, context_token_start_index - 1, -1):
                                             if offsets[token_idx][1] >= answer_end_char > offsets[token_idx][0] or \
                                                (offsets[token_idx][1] == answer_end_char and offsets[token_idx][0] < answer_end_char):
                                                 temp_end_pos = token_idx; end_token_found = True; break
                                        if start_token_found and end_token_found and temp_start_pos <= temp_end_pos:
                                             feature["start_positions"] = temp_start_pos; feature["end_positions"] = temp_end_pos
                                        else: local_debug_logger.debug(f"Token mapping failed L{line_num}, Ex:{example_id}")

                            total_features_generated += 1
                            yield feature # Yield feature

                    # --- End inner try-except ---
                    except json.JSONDecodeError: logging.warning(f"Skipping malformed JSON L{line_num} in {filepath}"); skipped_lines += 1
                    except MemoryError: logging.error(f"MemoryError L{line_num} in {filepath}. Skipping rest."); skipped_lines += (file_processed_lines - line_num + 1); raise StopIteration
                    except Exception as e: logging.error(f"Error L{line_num} in {filepath}: {e}"); skipped_lines += 1
            # --- End file loop ---
        # --- End outer try-except ---
        except FileNotFoundError: logging.error(f"File not found: {filepath}")
        except Exception as e: logging.error(f"Error opening/reading {filepath}: {e}")

    # --- Use generator to create and save dataset ---
    try:
        include_token_type_ids = "token_type_ids" in tokenizer.model_input_names
        feature_schema_dict = {
            'example_id': datasets.Value('string'),
            'input_ids': datasets.Sequence(datasets.Value('int32')),
            'attention_mask': datasets.Sequence(datasets.Value('int8')),
            'start_positions': datasets.Value('int32'),
            'end_positions': datasets.Value('int32'),
            'offset_mapping': datasets.Sequence(datasets.Sequence(datasets.Value('int32'), length=2), length=-1)
        }
        if include_token_type_ids: feature_schema_dict['token_type_ids'] = datasets.Sequence(datasets.Value('int8'))
        ds_features = datasets.Features(feature_schema_dict)

        logging.debug(f"Creating dataset from generator for {filepath}")
        hf_dataset = datasets.Dataset.from_generator(feature_generator, features=ds_features)

        if len(hf_dataset) > 0:
            temp_filename = f"features_{uuid.uuid4()}.arrow"
            output_arrow_path = os.path.join(temp_dir, temp_filename)
            # Ensure parent directory exists before saving
            os.makedirs(os.path.dirname(output_arrow_path), exist_ok=True)
            hf_dataset.save_to_disk(output_arrow_path)
            logging.debug(f"Saved temporary features for {filepath} to {output_arrow_path}")
            total_features_generated = len(hf_dataset)
        else:
            logging.info(f"No features yielded for {filepath}. Skipping temp file.")
            total_features_generated = 0

    except Exception as e:
        logging.error(f"Error creating/saving dataset from generator for {filepath}: {e}", exc_info=True)
        return None

    logging.info(f"Finished {filepath}: gen {total_features_generated} features, saved to {output_arrow_path if output_arrow_path else 'N/A'}. Skipped {skipped_lines} lines.")
    return output_arrow_path


# --- Main Execution Function ---
def run_preprocessing(source_dir, final_target_dir, split_name):
    """ Finds files, runs preprocessing in parallel saving temp files, and merges them to final dir. """
    logging.info(f"Starting preprocessing for split: {split_name}")

    # <<< Resolve and check source directory path more robustly >>>
    source_accessible = False
    abs_source_dir = ""
    try:
        # Convert to absolute path first
        abs_source_dir = os.path.abspath(source_dir)
        logging.info(f"Attempting to access source directory: {abs_source_dir}")
        # Use pathlib for potentially better symlink/shortcut handling
        source_path = pathlib.Path(abs_source_dir)

        # --- MODIFIED CHECK: Try os.listdir first ---
        try:
            logging.info(f"Attempting to list contents of: {abs_source_dir}")
            contents = os.listdir(abs_source_dir)
            logging.info(f"Successfully listed contents (found {len(contents)} items). Assuming directory is accessible.")
            source_accessible = True
        except FileNotFoundError:
             logging.error(f"Source directory not found using os.listdir: {abs_source_dir}")
        except PermissionError:
             logging.error(f"Permission denied trying to list directory: {abs_source_dir}")
        except Exception as list_e:
             logging.error(f"Unexpected error listing directory {abs_source_dir}: {list_e}")

        # Fallback check if listdir failed, or just log the is_dir result
        is_dir = source_path.is_dir()
        logging.info(f"Result of source_path.is_dir(): {is_dir}")
        if not source_accessible and not is_dir: # If both listdir failed and is_dir is False
             logging.error(f"Directory check failed for: {abs_source_dir}")
             return # Exit if not accessible
        elif not source_accessible and is_dir:
             logging.warning(f"os.path.isdir() is True but os.listdir() failed for {abs_source_dir}. Proceeding cautiously.")
             source_accessible = True # Assume accessible based on is_dir

        if source_accessible:
             logging.info(f"Successfully accessed source directory: {abs_source_dir}")

    except Exception as path_e:
        logging.error(f"Error checking source directory path {source_dir}: {path_e}")
        return
    # <<< End of path check >>>

    if not source_accessible:
         logging.error(f"Could not confirm accessibility of source directory: {abs_source_dir}. Exiting.")
         return

    logging.info(f"Final Target directory: {final_target_dir}")

    # Use the confirmed absolute path
    files_to_process = [os.path.join(abs_source_dir, f) for f in os.listdir(abs_source_dir) if f.lower().endswith('.jsonl')]
    if not files_to_process:
        logging.warning(f"No .jsonl files found in {abs_source_dir}."); return

    logging.info(f"Found {len(files_to_process)} '.jsonl' files for '{split_name}'.")

    # <<< *** UPDATED: Create temp dir inside TEMP_PROCESSING_BASE_DIR *** >>>
    # Create a temporary directory for this split's intermediate files
    # The base directory is now set in the config section
    temp_dir_split = tempfile.mkdtemp(prefix=f"nq_proc_{split_name}_", dir=TEMP_PROCESSING_BASE_DIR)
    logging.info(f"Created temporary directory for intermediate files: {temp_dir_split}")
    # <<< *** END UPDATE *** >>>

    # --- Parallel Processing ---
    num_processes = multiprocessing.cpu_count() - 1 if NUM_PROCESSES is None and multiprocessing.cpu_count() > 1 else (NUM_PROCESSES if NUM_PROCESSES else 1)
    num_processes = max(1, num_processes - 1) if num_processes > 1 else 1 # Leave one core free
    logging.info(f"Using {num_processes} worker processes.")

    process_func = partial(preprocess_and_save_single_file, tokenizer_name=TOKENIZER_CHECKPOINT,
                           max_seq_len=MAX_SEQ_LENGTH, doc_stride=DOC_STRIDE, temp_dir=temp_dir_split)

    temp_file_paths = []
    total_start_time = time.time()

    try: # Run the pool
        with multiprocessing.Pool(processes=num_processes) as pool:
            results_iterator = pool.imap_unordered(process_func, files_to_process)
            logging.info(f"Processing {len(files_to_process)} files...")
            for temp_path in tqdm(results_iterator, total=len(files_to_process), desc=f"Preprocessing {split_name} files"):
                if temp_path and os.path.exists(temp_path):
                    try: # Check if temp dataset is valid and non-empty
                        # Add a small delay before loading, might help with sync issues
                        time.sleep(0.1)
                        d = datasets.load_from_disk(temp_path)
                        if len(d) > 0: temp_file_paths.append(temp_path)
                        else:
                            logging.warning(f"Temp dataset {temp_path} empty. Skipping & deleting.")
                            try: shutil.rmtree(temp_path)
                            except Exception as del_err: logging.error(f"Error deleting empty temp dir {temp_path}: {del_err}")
                    except Exception as load_err: logging.error(f"Failed load/check temp dataset {temp_path}: {load_err}. Skipping.")
                elif temp_path: logging.warning(f"Worker returned {temp_path} but it doesn't exist.")
                else: logging.warning(f"Worker returned invalid path: {temp_path}.")
    except Exception as e:
        logging.error(f"Parallel processing error: {e}", exc_info=True)
        # Attempt cleanup even on error
        try: shutil.rmtree(temp_dir_split); logging.info(f"Cleaned up temp dir after error: {temp_dir_split}")
        except Exception as cleanup_e: logging.error(f"Error cleaning temp dir {temp_dir_split} after error: {cleanup_e}")
        return # Stop processing this split

    total_end_time = time.time()
    logging.info(f"Parallel processing finished in {total_end_time - total_start_time:.2f} seconds.")
    logging.info(f"Generated {len(temp_file_paths)} non-empty temporary feature files.")

    if not temp_file_paths:
        logging.error(f"No non-empty temporary files generated for {split_name}. Check logs.")
        # Attempt cleanup
        try: shutil.rmtree(temp_dir_split); logging.info(f"Cleaned up empty temp dir: {temp_dir_split}")
        except Exception as cleanup_e: logging.error(f"Error cleaning temp dir {temp_dir_split}: {cleanup_e}")
        return

    # --- Merge Temporary Datasets and Save Safely ---
    final_split_output_path = os.path.join(final_target_dir, split_name)
    temp_final_save_path = final_split_output_path + f"_temp_{uuid.uuid4()}"
    logging.info(f"Merging temporary datasets into: {temp_final_save_path}")

    merged_successfully = False
    try:
        logging.info(f"Loading {len(temp_file_paths)} datasets...")
        # Explicitly load datasets into a list first before concatenating
        # This ensures handles might be released before saving the final one.
        list_of_datasets = []
        for path in tqdm(temp_file_paths, desc="Loading temp datasets"):
             list_of_datasets.append(datasets.load_from_disk(path))

        logging.info("Concatenating datasets...")
        final_dataset = datasets.concatenate_datasets(list_of_datasets)
        del list_of_datasets # Try to release memory/handles

        logging.info(f"Final dataset for '{split_name}' has {len(final_dataset)} features.")

        logging.info(f"Saving merged dataset temporarily to {temp_final_save_path}...")
        final_dataset.save_to_disk(temp_final_save_path)
        logging.info(f"Temporary save complete.")
        merged_successfully = True # Mark success only after save_to_disk completes

        # --- Atomic Replace (as much as possible) ---
        logging.info(f"Replacing existing target directory (if any) with new data: {final_split_output_path}")
        if os.path.exists(final_split_output_path):
            logging.warning(f"Removing existing final dataset directory: {final_split_output_path}")
            shutil.rmtree(final_split_output_path)

        logging.info(f"Renaming {temp_final_save_path} to {final_split_output_path}")
        os.rename(temp_final_save_path, final_split_output_path)
        logging.info(f"Final dataset successfully saved to {final_split_output_path}.")

    except Exception as e:
        logging.error(f"Failed to merge or save final dataset for {split_name}: {e}", exc_info=True)
        # Attempt to clean up the temporary final save path if it exists and merge failed
        if os.path.exists(temp_final_save_path):
            try: shutil.rmtree(temp_final_save_path); logging.info(f"Cleaned up failed temporary save: {temp_final_save_path}")
            except Exception as cleanup_e: logging.error(f"Error cleaning failed temp save {temp_final_save_path}: {cleanup_e}")
    finally:
        # --- Clean up intermediate temporary directory ---
        # <<< Added Retry Logic for Cleanup >>>
        logging.info(f"Cleaning up intermediate temporary directory: {temp_dir_split}")
        max_retries = 5
        retry_delay = 2 # seconds
        for attempt in range(max_retries):
            try:
                # Add a small delay before attempting deletion
                time.sleep(retry_delay)
                shutil.rmtree(temp_dir_split)
                logging.info(f"Successfully cleaned up intermediate temporary directory on attempt {attempt + 1}.")
                break # Exit loop if successful
            except PermissionError as perm_err:
                 logging.warning(f"Cleanup attempt {attempt + 1}/{max_retries} failed (PermissionError): {perm_err}. Retrying...")
                 if attempt == max_retries - 1:
                     logging.error(f"Failed to clean up temporary directory {temp_dir_split} after {max_retries} attempts due to PermissionError. Please delete it manually.")
            except FileNotFoundError:
                 logging.warning(f"Temporary directory {temp_dir_split} not found during cleanup (already deleted?).")
                 break # Exit loop if not found
            except Exception as cleanup_e:
                logging.error(f"Cleanup attempt {attempt + 1}/{max_retries} failed with unexpected error: {cleanup_e}")
                if attempt == max_retries - 1:
                    logging.error(f"Failed to clean up temporary directory {temp_dir_split} after {max_retries} attempts. Please delete it manually.")
        # <<< End of Retry Logic >>>


if __name__ == "__main__":
    multiprocessing.freeze_support()
    logging.info("Starting NQ Preprocessing Script (Memory Optimized v3 - Safer Save + Path Handling)")

    # <<< Ensure BASE_DATA_DIR is set correctly above >>>
    if BASE_DATA_DIR == '.':
         logging.warning("BASE_DATA_DIR is set to '.'. Using current working directory. For robustness with links/shortcuts, please specify the full absolute path.")
         BASE_DATA_DIR = os.path.abspath('.') # Use absolute path of current dir if not set

    # Reconstruct paths using potentially updated BASE_DATA_DIR
    SOURCE_DATA_BASE = os.path.join(BASE_DATA_DIR, 'unarchived', 'v1.0')
    TRAIN_SOURCE_DIR = os.path.join(SOURCE_DATA_BASE, 'train')
    DEV_SOURCE_DIR = os.path.join(SOURCE_DATA_BASE, 'dev')
    FINAL_OUTPUT_DIR = os.path.join(BASE_DATA_DIR, 'nq_processed_final')
    # <<< *** Create the base temp dir if using local path *** >>>
    TEMP_PROCESSING_BASE_DIR = os.path.join(BASE_DATA_DIR, 'temp_preprocessing_files')
    os.makedirs(TEMP_PROCESSING_BASE_DIR, exist_ok=True)
    # <<< *** END *** >>>
    # Re-ensure final output dir exists after path reconstruction
    os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)
    # Reconfigure log file path in case BASE_DATA_DIR changed
    log_file_path = os.path.join(FINAL_OUTPUT_DIR, "preprocessing.log")
    # Find the FileHandler and update its path (a bit complex, might be easier to re-init logger)
    # Close and remove old file handlers before reconfiguring
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]: # Iterate over a copy
        if isinstance(handler, logging.FileHandler):
            handler.close()
            root_logger.removeHandler(handler)
    # Re-add file handler with potentially updated path
    root_logger.addHandler(logging.FileHandler(log_file_path, mode='w')) # Overwrite on start

    debug_logger = logging.getLogger('nq_debug')
    for handler in debug_logger.handlers[:]: # Iterate over a copy
         if isinstance(handler, logging.FileHandler):
             handler.close()
             debug_logger.removeHandler(handler)
    debug_logger.addHandler(logging.FileHandler(log_file_path, mode='a')) # Append debug messages


    logging.info(f"Using BASE_DATA_DIR: {BASE_DATA_DIR}")
    logging.info(f"Using TEMP_PROCESSING_BASE_DIR: {TEMP_PROCESSING_BASE_DIR}") # Log temp base dir
    logging.info(f"Log file location: {log_file_path}")


    # --- Preprocess Training Data ---
    # <<< Use run_preprocessing which now contains the check >>>
    run_preprocessing(TRAIN_SOURCE_DIR, FINAL_OUTPUT_DIR, 'train')

    # --- Preprocess Development Data ---
    # <<< Use run_preprocessing which now contains the check >>>
    run_preprocessing(DEV_SOURCE_DIR, FINAL_OUTPUT_DIR, 'dev')


    logging.info("NQ Preprocessing Script finished.")
    logging.info(f"Processed datasets saved in: {FINAL_OUTPUT_DIR}")
    logging.info(f"Check log file for details: {log_file_path}")

