import os
import json
import gzip
import re
from collections import Counter, defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns # For heatmap
from tqdm import tqdm
import logging
import multiprocessing
from functools import partial
import time
import sys # For exit

# --- Dependencies ---
try:
    from transformers import AutoTokenizer
except ImportError:
    print("Error: 'transformers' library not found. Please install it: pip install transformers torch")
    sys.exit(1)

# --- Configuration ---
BASE_DATA_DIR = '.' # Assume data folders are in the current directory
# <<< Using the unarchived data directory from previous step >>>
TRAIN_DIR = os.path.join(BASE_DATA_DIR, 'unarchived', 'v1.0', 'train')
DEV_DIR = os.path.join(BASE_DATA_DIR, 'unarchived', 'v1.0', 'dev')
# <<< Saving results to a new directory >>>
RESULTS_DIR = os.path.join(BASE_DATA_DIR, 'eda_results_enhanced')

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Setup logging
log_file_path = os.path.join(RESULTS_DIR, "eda_enhanced.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler() # Also print logs to console
    ]
)

# --- Tokenizer Initialization ---
# Using a common tokenizer. Adjust if needed for specific model comparisons later.
TOKENIZER_CHECKPOINT = "bert-base-uncased"
try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_CHECKPOINT)
    logging.info(f"Tokenizer '{TOKENIZER_CHECKPOINT}' loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load tokenizer '{TOKENIZER_CHECKPOINT}': {e}")
    logging.error("Please ensure 'transformers' and a backend (torch/tensorflow) are installed.")
    sys.exit(1)


# --- Histogram Bins ---
QUESTION_LEN_BINS = np.linspace(0, 300, 61) # Chars
QUESTION_TOKEN_LEN_BINS = np.linspace(0, 100, 51) # Tokens

DOC_HTML_LEN_BINS = np.logspace(2, 7, 51) # HTML Bytes
DOC_CLEANED_LEN_BINS = np.logspace(2, 6, 41) # Cleaned Chars (100 to 1M)

SHORT_ANSWER_COUNT_BINS = np.arange(0, 11, 1) # 0 to 10 answers
SHORT_ANSWER_CHAR_LEN_BINS = np.linspace(0, 200, 41) # Chars
SHORT_ANSWER_TOKEN_LEN_BINS = np.linspace(0, 50, 26) # Tokens
SHORT_ANSWER_REL_POS_BINS = np.linspace(0, 1, 21) # Relative position (0 to 1 in 20 bins)

LONG_ANSWER_LEN_BINS = np.logspace(1, 5, 41) # Bytes (10 to 100k)

# Plot settings
PLOT_DPI = 150
HEATMAP_MAX_TYPES = 15 # Limit heatmap categories for readability
HEATMAP_MAX_WORDS = 20

# --- Helper Functions ---

def clean_html_basic(html_content):
    if not html_content: return ""
    clean_text = re.sub(r'<[^>]+>', ' ', html_content)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text

def get_question_start_word(question_text):
    if not question_text: return "N/A"
    match = re.match(r'^\s*(\w+)', question_text)
    return match.group(1).lower() if match else "N/A"

def get_answerability_type(annotation):
    types = []
    has_short = False
    # Check for valid short answers
    if annotation.get('short_answers'):
        for ans in annotation['short_answers']:
            # Ensure start/end exist and start <= end (NQ end is exclusive, so < okay)
            if ('start_byte' in ans and 'end_byte' in ans and
                ans['start_byte'] >= 0 and ans['end_byte'] >= 0 and # Non-negative
                ans['start_byte'] < ans['end_byte']):
                 has_short = True
                 break
    if has_short:
         types.append("short")

    # Check for valid long answer span
    long_answer = annotation.get('long_answer', {})
    if (long_answer and long_answer.get('start_byte', -1) != -1 and
        long_answer.get('end_byte', -1) != -1 and
        long_answer['start_byte'] < long_answer['end_byte']):
         types.append("long")

    # Check Yes/No answer
    yes_no_answer = annotation.get('yes_no_answer', 0) # NQ: 0=NONE, 1=YES, 2=NO
    if yes_no_answer == 1:
        types.append("yes")
    elif yes_no_answer == 2:
        types.append("no")

    if not types:
        return "unanswerable"
    else:
        return "_".join(sorted(types)) # Combine types like "short_long", "yes", etc.

def update_histogram(value, bins, hist_counter):
    """ Finds the correct bin and increments the counter """
    # np.digitize returns indices from 1 to len(bins).
    # We map index k to interval bins[k-1] <= value < bins[k]
    bin_index = np.digitize(value, bins)
    # Clamp index to handle edge cases (values exactly matching last bin edge or outside range)
    # Ensure index is at least 1 and at most len(bins) - 1
    clamped_index = max(1, min(bin_index, len(bins) - 1))
    hist_counter[clamped_index] += 1

# --- Plotting Functions ---

def plot_histogram(hist_counter, bins, title, xlabel, filename, log_scale='auto'):
    if not hist_counter:
        logging.warning(f"No data to plot for: {title}")
        return

    bin_centers = (bins[:-1] + bins[1:]) / 2
    counts = [hist_counter.get(i + 1, 0) for i in range(len(bins) - 1)]

    if sum(counts) == 0:
        logging.warning(f"All counts are zero for histogram: {title}")
        return

    plt.figure(figsize=(10, 6))
    plt.bar(bin_centers, counts, width=np.diff(bins), align='center', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")

    use_log = False
    if log_scale == True:
         use_log = True
    elif log_scale == 'auto':
         # Heuristic: use log if max count is much larger than median non-zero count
         non_zero_counts = [c for c in counts if c > 0]
         if non_zero_counts:
             max_c = np.max(non_zero_counts)
             median_c = np.median(non_zero_counts)
             if max_c > 100 * median_c and max_c > 10: # Adjust multiplier as needed
                 use_log = True
         # Also use log if bin range is very wide (like doc lengths)
         if not use_log and bins[0] > 0 and np.log10(bins[-1] / bins[0]) > 3:
             use_log = True

    if use_log:
        plt.yscale('log')
        plt.ylabel("Frequency (log scale)")

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    try:
        plt.savefig(filename, dpi=PLOT_DPI)
        logging.info(f"Saved plot: {filename}")
    except Exception as e:
        logging.error(f"Failed to save plot {filename}: {e}")
    plt.close()


def plot_bar_chart(counter, title, xlabel, filename, top_n=20):
    if not counter:
        logging.warning(f"No data to plot for: {title}")
        return

    if top_n is None or top_n >= len(counter):
        items_to_plot = counter.most_common()
        plot_title = f"{title} (All)"
    else:
        items_to_plot = counter.most_common(top_n)
        plot_title = f"{title} (Top {top_n})"

    if not items_to_plot:
         logging.warning(f"No items to plot for {title}")
         return

    labels, values = zip(*items_to_plot)

    plt.figure(figsize=(12, max(7, len(labels) // 2)))
    plt.bar(labels, values, color='skyblue', edgecolor='black')
    plt.title(plot_title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    try:
        plt.savefig(filename, dpi=PLOT_DPI)
        logging.info(f"Saved plot: {filename}")
    except Exception as e:
        logging.error(f"Failed to save plot {filename}: {e}")
    plt.close()


def plot_heatmap(data_matrix, x_labels, y_labels, title, filename):
    """ Plots a heatmap from a pandas DataFrame """
    if data_matrix.empty:
        logging.warning(f"No data to plot heatmap for: {title}")
        return

    plt.figure(figsize=(min(20, len(x_labels)), min(15, len(y_labels)))) # Adjust size
    sns.heatmap(data_matrix, annot=True, fmt=".0f", cmap="viridis", linewidths=.5)
    plt.title(title)
    plt.xlabel("Answerability Type")
    plt.ylabel("Question Starting Word")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    try:
        plt.savefig(filename, dpi=PLOT_DPI)
        logging.info(f"Saved plot: {filename}")
    except Exception as e:
        logging.error(f"Failed to save heatmap plot {filename}: {e}")
    plt.close()


# --- Process Single File Function ---
def process_single_file_eda(filepath, tokenizer_name):
    """ Processes a single data file and returns its EDA results """
    # Initialize tokenizer within the process if needed, or ensure it's picklable
    # For HuggingFace tokenizers, it's often better to pass the name and load here
    try:
        local_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        logging.error(f"Worker process failed to load tokenizer '{tokenizer_name}': {e}")
        return None # Cannot proceed without tokenizer

    # Initialize local counters for this file
    results = {
        'count': 0,
        'q_start_words': Counter(),
        'ans_types': Counter(),
        'q_char_len_hist': Counter(),
        'q_token_len_hist': Counter(),
        'doc_html_len_hist': Counter(),
        'doc_cleaned_len_hist': Counter(),
        'sa_count_hist': Counter(),
        'sa_char_len_hist': Counter(),
        'sa_token_len_hist': Counter(),
        'sa_rel_pos_hist': Counter(),
        'la_len_hist': Counter(),
        'yes_no_counts': Counter(), # For explicit YES/NO counts
        'type_vs_answerability': defaultdict(Counter) # Nested counter
    }

    # Use the correct file extension based on the input path
    # Assuming input files are .jsonl (already decompressed)
    open_func = open # Not gzip.open

    try:
        with open_func(filepath, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    example = json.loads(line)
                    results['count'] += 1

                    question_text = example.get('question_text', '')
                    document_html = example.get('document', {}).get('html', '')
                    # NQ annotations are list, usually take first one for basic EDA
                    # More complex analysis might look at all annotations if multiple exist
                    annotation = example.get('annotations', [{}])[0]

                    # --- Basic Info ---
                    q_start_word = get_question_start_word(question_text)
                    results['q_start_words'][q_start_word] += 1

                    ans_type = get_answerability_type(annotation)
                    results['ans_types'][ans_type] += 1

                    # --- Question Analysis ---
                    q_char_len = len(question_text)
                    update_histogram(q_char_len, QUESTION_LEN_BINS, results['q_char_len_hist'])
                    # Tokenize question (handle potential errors)
                    try:
                        q_tokens = local_tokenizer.encode(question_text, add_special_tokens=True)
                        q_token_len = len(q_tokens)
                        update_histogram(q_token_len, QUESTION_TOKEN_LEN_BINS, results['q_token_len_hist'])
                    except Exception as token_err:
                        logging.warning(f"Tokenization error for question in file {filepath}, line approx {results['count']}: {token_err}")


                    # --- Document Analysis ---
                    doc_html_len = len(document_html.encode('utf-8')) # Bytes
                    update_histogram(doc_html_len, DOC_HTML_LEN_BINS, results['doc_html_len_hist'])
                    # Cleaned document length
                    cleaned_doc = clean_html_basic(document_html)
                    doc_cleaned_len = len(cleaned_doc)
                    update_histogram(doc_cleaned_len, DOC_CLEANED_LEN_BINS, results['doc_cleaned_len_hist'])
                    # Note: Tokenizing full cleaned docs here would be very slow. Skipped for now.

                    # --- Answer Analysis ---
                    short_answers = annotation.get('short_answers', [])
                    valid_short_answers = []
                    if short_answers: # Check if list exists and is not empty
                         for ans in short_answers:
                             # Check validity again
                             if ('start_byte' in ans and 'end_byte' in ans and
                                 ans['start_byte'] >= 0 and ans['end_byte'] >= 0 and
                                 ans['start_byte'] < ans['end_byte']):
                                  valid_short_answers.append(ans)

                    if valid_short_answers:
                        # Short Answer Count
                        sa_count = len(valid_short_answers)
                        update_histogram(sa_count, SHORT_ANSWER_COUNT_BINS, results['sa_count_hist'])

                        for short_ans in valid_short_answers:
                            start_byte = short_ans['start_byte']
                            end_byte = short_ans['end_byte']

                            # Short Answer Length (Bytes/Chars)
                            sa_byte_len = end_byte - start_byte
                            update_histogram(sa_byte_len, SHORT_ANSWER_CHAR_LEN_BINS, results['sa_char_len_hist']) # Using char bins for byte length too

                            # Extract text for tokenization (handle potential errors)
                            try:
                                # Decode bytes from HTML using start/end bytes
                                sa_text = document_html.encode('utf-8')[start_byte:end_byte].decode('utf-8', errors='ignore')
                                sa_tokens = local_tokenizer.encode(sa_text, add_special_tokens=False) # Don't add CLS/SEP
                                sa_token_len = len(sa_tokens)
                                update_histogram(sa_token_len, SHORT_ANSWER_TOKEN_LEN_BINS, results['sa_token_len_hist'])
                            except Exception as decode_token_err:
                                logging.warning(f"Decode/Tokenize error for SA bytes {start_byte}-{end_byte} in {filepath}, line approx {results['count']}: {decode_token_err}")


                            # Short Answer Relative Position
                            if doc_html_len > 0:
                                rel_pos = start_byte / doc_html_len
                                update_histogram(rel_pos, SHORT_ANSWER_REL_POS_BINS, results['sa_rel_pos_hist'])

                    # Long Answer Length
                    long_answer = annotation.get('long_answer', {})
                    if (long_answer and long_answer.get('start_byte', -1) != -1 and
                        long_answer.get('end_byte', -1) != -1 and
                        long_answer['start_byte'] < long_answer['end_byte']):
                        la_len = long_answer['end_byte'] - long_answer['start_byte']
                        update_histogram(la_len, LONG_ANSWER_LEN_BINS, results['la_len_hist'])

                    # Yes/No Counts
                    yes_no_answer = annotation.get('yes_no_answer', 0)
                    if yes_no_answer == 1:
                        results['yes_no_counts']['YES'] += 1
                    elif yes_no_answer == 2:
                        results['yes_no_counts']['NO'] += 1

                    # --- Cross Analysis ---
                    results['type_vs_answerability'][q_start_word][ans_type] += 1

                except json.JSONDecodeError:
                    logging.warning(f"Skipping malformed JSON line in {filepath}, line approx {results['count']+1}")
                except Exception as e:
                    logging.error(f"Error processing line in {filepath}, approx line {results['count']+1}: {e}", exc_info=False) # Disable full traceback spam
    except FileNotFoundError:
         logging.error(f"File not found by worker process: {filepath}")
         return None
    except Exception as e:
        logging.error(f"Error opening or reading file {filepath}: {e}")
        return None # Return None if file processing failed

    return results


# --- Main Processing Function ---
def run_parallel_eda(data_dir, results_dir, split_name):
    """ Processes all files in a directory in parallel for EDA """
    logging.info(f"Starting Enhanced Parallel EDA for split: {split_name} in directory: {data_dir}")

    if not os.path.isdir(data_dir):
        logging.error(f"Data directory not found: {data_dir}")
        return

    # Find all .jsonl files (assuming decompressed)
    files_to_process = [os.path.join(data_dir, f)
                        for f in os.listdir(data_dir)
                        if f.lower().endswith('.jsonl')] # Look for .jsonl

    if not files_to_process:
        logging.warning(f"No .jsonl files found in {data_dir}. Ensure data is decompressed.")
        return

    # Initialize aggregate results dictionary
    aggregate_results = {
        'count': 0,
        'q_start_words': Counter(),
        'ans_types': Counter(),
        'q_char_len_hist': Counter(),
        'q_token_len_hist': Counter(),
        'doc_html_len_hist': Counter(),
        'doc_cleaned_len_hist': Counter(),
        'sa_count_hist': Counter(),
        'sa_char_len_hist': Counter(),
        'sa_token_len_hist': Counter(),
        'sa_rel_pos_hist': Counter(),
        'la_len_hist': Counter(),
        'yes_no_counts': Counter(),
        'type_vs_answerability': defaultdict(Counter)
    }

    # --- Parallel Processing ---
    num_processes = multiprocessing.cpu_count() - 1 if multiprocessing.cpu_count() > 1 else 1
    logging.info(f"Using {num_processes} worker processes.")

    # Use partial to pass the tokenizer name to the worker function
    process_func = partial(process_single_file_eda, tokenizer_name=TOKENIZER_CHECKPOINT)

    try:
        with multiprocessing.Pool(processes=num_processes) as pool:
            results_iterator = pool.imap_unordered(process_func, files_to_process)
            logging.info(f"Processing {len(files_to_process)} files...")
            for result in tqdm(results_iterator, total=len(files_to_process), desc=f"Aggregating {split_name} results"):
                if result: # Aggregate results if the file was processed successfully
                    aggregate_results['count'] += result['count']
                    # Update all counters and defaultdicts
                    for key, value in result.items():
                        if isinstance(value, Counter):
                            aggregate_results[key].update(value)
                        elif isinstance(value, defaultdict):
                            for sub_key, sub_counter in value.items():
                                aggregate_results[key][sub_key].update(sub_counter)
                    # Note: 'count' is handled above
                else:
                    logging.warning("Received None result from a worker process, likely due to file or tokenizer error.")

    except Exception as e:
        logging.error(f"An error occurred during parallel processing: {e}", exc_info=True)
        return # Stop if the pool encounters a major issue

    logging.info(f"Finished processing. Aggregated {aggregate_results['count']} examples for {split_name}.")
    if aggregate_results['count'] == 0:
        logging.warning(f"No examples were successfully processed for {split_name}. Skipping result generation.")
        return

    # --- Save Aggregated Results ---
    summary_file = os.path.join(results_dir, f"{split_name}_eda_summary_enhanced.txt")
    try:
        with open(summary_file, 'w', encoding='utf-8') as f:
            total_examples = aggregate_results['count']
            f.write(f"--- Enhanced EDA Summary for {split_name} ---\n\n")
            f.write(f"Total Examples Processed: {total_examples}\n\n")

            f.write("Answerability Distribution:\n")
            for type_name, count in aggregate_results['ans_types'].most_common():
                percentage = (count / total_examples) * 100 if total_examples else 0
                f.write(f"  - {type_name}: {count} ({percentage:.2f}%)\n")
            f.write("\n")

            f.write("Yes/No Answer Counts:\n")
            yes_count = aggregate_results['yes_no_counts'].get('YES', 0)
            no_count = aggregate_results['yes_no_counts'].get('NO', 0)
            f.write(f"  - YES: {yes_count} ({(yes_count / total_examples) * 100:.2f}%)\n")
            f.write(f"  - NO: {no_count} ({(no_count / total_examples) * 100:.2f}%)\n")
            f.write("\n")

            f.write("Top 20 Question Starting Words:\n")
            for word, count in aggregate_results['q_start_words'].most_common(20):
                percentage = (count / total_examples) * 100 if total_examples else 0
                f.write(f"  - {word}: {count} ({percentage:.2f}%)\n")
            f.write("\n")

            # Add more summary stats if desired (e.g., avg lengths)

        logging.info(f"Saved summary statistics to: {summary_file}")
    except Exception as e:
        logging.error(f"Failed to write summary file {summary_file}: {e}")


    # --- Generate Plots ---
    # Question Lengths
    plot_histogram(aggregate_results['q_char_len_hist'], QUESTION_LEN_BINS, f'{split_name.capitalize()} Question Length (Chars)', 'Characters', os.path.join(results_dir, f'{split_name}_question_char_length_hist.png'))
    plot_histogram(aggregate_results['q_token_len_hist'], QUESTION_TOKEN_LEN_BINS, f'{split_name.capitalize()} Question Length (Tokens)', 'Tokens', os.path.join(results_dir, f'{split_name}_question_token_length_hist.png'))

    # Document Lengths
    plot_histogram(aggregate_results['doc_html_len_hist'], DOC_HTML_LEN_BINS, f'{split_name.capitalize()} Document HTML Length (Bytes)', 'Bytes', os.path.join(results_dir, f'{split_name}_document_html_length_hist.png'), log_scale=True)
    plot_histogram(aggregate_results['doc_cleaned_len_hist'], DOC_CLEANED_LEN_BINS, f'{split_name.capitalize()} Document Cleaned Length (Chars)', 'Characters', os.path.join(results_dir, f'{split_name}_document_cleaned_length_hist.png'), log_scale=True)

    # Short Answer Stats
    plot_histogram(aggregate_results['sa_count_hist'], SHORT_ANSWER_COUNT_BINS, f'{split_name.capitalize()} Short Answers per Question', '# Short Answers', os.path.join(results_dir, f'{split_name}_short_answer_count_hist.png'))
    plot_histogram(aggregate_results['sa_char_len_hist'], SHORT_ANSWER_CHAR_LEN_BINS, f'{split_name.capitalize()} Short Answer Length (Bytes)', 'Bytes', os.path.join(results_dir, f'{split_name}_short_answer_byte_length_hist.png'))
    plot_histogram(aggregate_results['sa_token_len_hist'], SHORT_ANSWER_TOKEN_LEN_BINS, f'{split_name.capitalize()} Short Answer Length (Tokens)', 'Tokens', os.path.join(results_dir, f'{split_name}_short_answer_token_length_hist.png'))
    plot_histogram(aggregate_results['sa_rel_pos_hist'], SHORT_ANSWER_REL_POS_BINS, f'{split_name.capitalize()} Short Answer Relative Start Position', 'Relative Position (Start Byte / HTML Bytes)', os.path.join(results_dir, f'{split_name}_short_answer_rel_pos_hist.png'))

    # Long Answer Length
    plot_histogram(aggregate_results['la_len_hist'], LONG_ANSWER_LEN_BINS, f'{split_name.capitalize()} Long Answer Length (Bytes)', 'Bytes', os.path.join(results_dir, f'{split_name}_long_answer_length_hist.png'), log_scale=True)

    # Bar Charts
    plot_bar_chart(aggregate_results['q_start_words'], f'{split_name.capitalize()} Question Starting Words', 'Starting Word', os.path.join(results_dir, f'{split_name}_question_start_words_bar.png'), top_n=HEATMAP_MAX_WORDS)
    plot_bar_chart(aggregate_results['ans_types'], f'{split_name.capitalize()} Answerability Types', 'Answer Type Combination', os.path.join(results_dir, f'{split_name}_answerability_types_bar.png'), top_n=HEATMAP_MAX_TYPES)

    # Heatmap: Question Type vs Answerability
    try:
        # Convert nested dict to DataFrame for heatmap
        q_vs_ans = aggregate_results['type_vs_answerability']
        # Limit categories for readability
        top_words = [w for w, _ in aggregate_results['q_start_words'].most_common(HEATMAP_MAX_WORDS)]
        top_types = [t for t, _ in aggregate_results['ans_types'].most_common(HEATMAP_MAX_TYPES)]

        # Filter data for top categories
        heatmap_data = defaultdict(lambda: defaultdict(int))
        for word, type_counts in q_vs_ans.items():
            if word in top_words:
                for ans_type, count in type_counts.items():
                    if ans_type in top_types:
                        heatmap_data[word][ans_type] = count

        # Convert to DataFrame, ensuring all top categories are present
        df_heatmap = pd.DataFrame(heatmap_data).fillna(0).astype(int)
        # Reindex to ensure consistent order and inclusion of all top categories
        df_heatmap = df_heatmap.reindex(index=top_words, columns=top_types, fill_value=0)


        if not df_heatmap.empty:
            plot_heatmap(
                df_heatmap,
                df_heatmap.columns, # Answer types
                df_heatmap.index,   # Question words
                f'{split_name.capitalize()} Question Start Word vs. Answerability Type',
                os.path.join(results_dir, f'{split_name}_qtype_vs_answerability_heatmap.png')
            )
        else:
             logging.warning("No data available for heatmap after filtering top categories.")

    except Exception as e:
        logging.error(f"Failed to generate heatmap plot: {e}", exc_info=True)


# --- Main Execution ---
if __name__ == "__main__":
    multiprocessing.freeze_support()
    logging.info("Starting Enhanced Parallel NQ EDA Script")
    start_time = pd.Timestamp.now()

    # --- Check Directories ---
    if not os.path.isdir(TRAIN_DIR):
         logging.error(f"Training data directory not found: {TRAIN_DIR}")
         logging.error("Please ensure the data has been decompressed into the 'unarchived' directory structure.")
         sys.exit(1)
    if not os.path.isdir(DEV_DIR):
         logging.warning(f"Development data directory not found: {DEV_DIR}. Skipping dev set.")


    # Run EDA for training set
    run_parallel_eda(TRAIN_DIR, RESULTS_DIR, 'train')

    # Run EDA for development/validation set
    if os.path.exists(DEV_DIR):
        run_parallel_eda(DEV_DIR, RESULTS_DIR, 'dev')
    else:
        logging.warning(f"Dev directory not found: {DEV_DIR}. Skipping.")

    end_time = pd.Timestamp.now()
    logging.info(f"Enhanced Parallel NQ EDA Script finished. Results saved to '{RESULTS_DIR}'. Total time: {end_time - start_time}")
