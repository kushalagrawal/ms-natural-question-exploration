import os
import logging
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
import collections

import numpy as np
import torch # Ensure torch is installed for transformers
import datasets
import evaluate # For metrics
from datasets import load_from_disk
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments, # Still used for some eval settings
    Trainer,
    default_data_collator,
)
from tqdm.auto import tqdm

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)


# --- Configuration Arguments ---
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )

@dataclass
class DataArguments:
    processed_eval_dataset_dir: str = field( # Changed from processed_dataset_dir
        metadata={"help": "Path to the directory containing the preprocessed Arrow dev/eval dataset."}
    )
    original_eval_data_path: str = field( # Path to original .jsonl file(s) for eval
        metadata={"help": "Path to the original NQ dev/eval .jsonl file or directory of .jsonl files."}
    )
    max_seq_length: int = field(
        default=384,
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )
    doc_stride: int = field( # Not strictly needed for eval if features are pre-generated but good for consistency
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    max_answer_length: int = field(
        default=30, # Max length of an answer to predict
        metadata={"help": "The maximum length of an answer that can be generated."}
    )
    n_best_size: int = field(
        default=20, # Number of top predictions to consider
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    null_score_diff_threshold: float = field(
        default=0.0, # SQuAD v2.0 default. May need tuning if model is SQuAD-tuned.
        metadata={"help": "Threshold for predicting 'no answer'. Lower value means more likely to predict an answer."}
    )
    output_prediction_file: Optional[str] = field(
        default=None, metadata={"help": "If provided, the path to save predictions."}
    )
    output_nbest_file: Optional[str] = field(
        default=None, metadata={"help": "If provided, the path to save nbest predictions."}
    )
    # <<< NEW: Argument to load original data for evaluation context/references >>>
    original_dev_path: Optional[str] = field( # Renamed for clarity
        default=None, metadata={"help": "Path to the original NQ dev .jsonl file (required for accurate evaluation)."}
    )


# --- Main Script ---
def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    # Example:
    # python your_eval_script.py --model_name_or_path deepset/roberta-base-squad2 \
    # --processed_eval_dataset_dir ./nq_processed_final/dev \
    # --original_eval_data_path ./unarchived/v1.0/dev \
    # --output_dir ./eval_results_roberta_squad2 \
    # --do_eval \
    # --per_device_eval_batch_size 16

    # For this example, define directly (replace with HfArgumentParser for cmd line)
    model_args = ModelArguments(
        #model_name_or_path="deepset/roberta-base-squad2" # Example: A SQuAD2-tuned model
        model_name_or_path="bert-large-uncased-whole-word-masking-finetuned-squad"
    )
    data_args = DataArguments(
        processed_eval_dataset_dir="./nq_processed_final/dev", # Path to your preprocessed dev set
        original_eval_data_path="./unarchived/v1.0/dev", # Path to original NQ dev .jsonl file(s)
        max_seq_length=384, # Should match your preprocessing
        doc_stride=128,     # Should match your preprocessing
        null_score_diff_threshold=0.0, # Default, can be experimented with
        original_dev_path="./unarchived/v1.0/dev" # Ensure this is set
    )
    # TrainingArguments is still used for output_dir, device settings, etc.
    # do_train will be implicitly False if not set or handled.
    training_args = TrainingArguments(
        output_dir=f"./eval_results_{model_args.model_name_or_path.replace('/', '_')}",
        do_eval=True, # Crucial: only perform evaluation
        do_train=False,
        per_device_eval_batch_size=16, # Adjust based on GPU memory
        logging_dir=f'./logs_eval_{model_args.model_name_or_path.replace("/", "_")}',
        report_to="none",
        # fp16=True, # Enable for faster inference if GPU supports it
    )

    # --- Argument Validation ---
    if not data_args.original_eval_data_path or not os.path.exists(data_args.original_eval_data_path): # Corrected from original_dev_path
        logger.error(f"Original evaluation data path not found or not specified: {data_args.original_eval_data_path}")
        return
    if not data_args.processed_eval_dataset_dir or not os.path.exists(data_args.processed_eval_dataset_dir):
        logger.error(f"Processed evaluation dataset directory not found: {data_args.processed_eval_dataset_dir}")
        return


    logger.info("-------------------- Evaluation Configuration --------------------")
    logger.info(f"Model Arguments: {model_args}")
    logger.info(f"Data Arguments: {data_args}")
    logger.info(f"Evaluation Output Dir: {training_args.output_dir}")
    logger.info("----------------------------------------------------------------")

    # --- Load Tokenizer and Model ---
    logger.info(f"Loading tokenizer: {model_args.tokenizer_name or model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading model: {model_args.model_name_or_path}")
    model_config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path, config=model_config, cache_dir=model_args.cache_dir
    )
    if tokenizer.pad_token_id == tokenizer.eos_token_id and hasattr(model, 'resize_token_embeddings'):
         model.resize_token_embeddings(len(tokenizer))

    # --- Load Preprocessed Evaluation Dataset ---
    logger.info(f"Loading preprocessed evaluation dataset from: {data_args.processed_eval_dataset_dir}")
    try:
        eval_dataset_features = load_from_disk(data_args.processed_eval_dataset_dir)
        logger.info(f"Preprocessed evaluation features loaded: {eval_dataset_features}")
        logger.info(f"Features in loaded dataset: {eval_dataset_features.column_names}") # Log column names
    except Exception as e:
        logger.error(f"Error loading preprocessed evaluation dataset: {e}")
        return

    # --- <<< MODIFIED: Validate ALL Input IDs before Evaluation >>> ---
    logger.info("Validating ALL input_ids in the evaluation dataset...")
    vocab_size = model.config.vocab_size
    logger.info(f"Model vocabulary size: {vocab_size}")
    invalid_ids_found = False

    # Iterate through the entire dataset
    for i in tqdm(range(len(eval_dataset_features)), desc="Validating input_ids"):
        try:
            input_ids = eval_dataset_features[i]["input_ids"]
            # Check if any ID is out of bounds
            if not all(0 <= token_id < vocab_size for token_id in input_ids):
                invalid_token = next((token_id for token_id in input_ids if not (0 <= token_id < vocab_size)), None)
                logger.error(f"Invalid input_id found in feature index {i} (example_id: {eval_dataset_features[i]['example_id']}).")
                logger.error(f"Invalid ID value: {invalid_token}, Vocab Size: {vocab_size}")
                logger.error(f"Problematic input_ids snippet: {input_ids[:20]}...{input_ids[-20:]}")
                invalid_ids_found = True
                break # Stop after first error found
        except Exception as e:
            logger.error(f"Error accessing data at index {i}: {e}")
            invalid_ids_found = True # Treat access error as a problem
            break

    if invalid_ids_found:
        logger.error("Invalid input_ids detected in preprocessed data. Evaluation cannot proceed.")
        logger.error("Please check the preprocessing script and ensure the correct tokenizer was used and data is not corrupted.")
        return # Stop the script
    else:
        logger.info(f"Validation of all input_ids passed.")
    # --- <<< END of Validation Check >>> ---


    # --- Load Original Evaluation Data (for context and ground truth answers) ---
    original_eval_examples_map = {} # Maps example_id to original data
    logger.info(f"Loading original evaluation data from: {data_args.original_eval_data_path}")
    try:
        if os.path.isdir(data_args.original_eval_data_path):
            original_files = [os.path.join(data_args.original_eval_data_path, f)
                              for f in os.listdir(data_args.original_eval_data_path) if f.endswith(".jsonl")]
        elif os.path.isfile(data_args.original_eval_data_path):
            original_files = [data_args.original_eval_data_path]
        else:
            logger.error(f"original_eval_data_path is not a valid file or directory: {data_args.original_eval_data_path}")
            return

        for filepath in original_files:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    example = json.loads(line)
                    example_id = str(example.get("example_id"))
                    if example_id:
                        original_eval_examples_map[example_id] = {
                            "context_html": example.get("document_html", ""), # Store original HTML
                            "annotations": example.get("annotations", [])
                        }
        logger.info(f"Loaded original data for {len(original_eval_examples_map)} evaluation examples.")
    except Exception as e:
        logger.error(f"Error loading original evaluation data: {e}")
        return

    # --- Prepare data for postprocessing (map features to original examples) ---
    eval_features_for_postprocessing = collections.defaultdict(lambda: {"features": []})
    # Add a global index to each feature for consistent mapping after prediction
    eval_dataset_features_indexed = eval_dataset_features.add_column(
        "global_feature_index", range(len(eval_dataset_features))
    )

    logger.info("Mapping evaluation features to original examples for postprocessing...")
    offset_mapping_missing = False # Flag to track if offset_mapping is missing
    for feature in tqdm(eval_dataset_features_indexed):
        example_id = feature["example_id"]
        if example_id in original_eval_examples_map:
            if "context_html" not in eval_features_for_postprocessing[example_id]: # Store context once
                eval_features_for_postprocessing[example_id]["context_html"] = original_eval_examples_map[example_id]["context_html"]
                eval_features_for_postprocessing[example_id]["annotations"] = original_eval_examples_map[example_id]["annotations"]

            # Check for offset_mapping
            if "offset_mapping" not in feature:
                if not offset_mapping_missing: # Log error only once
                     logger.error(f"FATAL: 'offset_mapping' key not found in preprocessed feature for example_id {example_id}. "
                                  f"This is required for postprocessing. Please ensure your preprocessing script saves it.")
                     offset_mapping_missing = True
                current_offset_mapping = [] # Placeholder to avoid crashing loop, but postprocessing will fail later
            else:
                current_offset_mapping = feature["offset_mapping"]

            eval_features_for_postprocessing[example_id]["features"].append({
                "offset_mapping": current_offset_mapping, # Use checked or placeholder offset_mapping
                "input_ids": feature["input_ids"],
                "token_type_ids": feature.get("token_type_ids"), # Handle if not present
                "global_feature_index": feature["global_feature_index"]
            })
        else:
            logger.warning(f"Example ID {example_id} from processed features not found in original data map. Skipping.")

    if offset_mapping_missing:
        logger.error("Evaluation cannot proceed because 'offset_mapping' is missing from the preprocessed data.")
        return # Stop if offset_mapping is missing

    logger.info("Finished mapping features for postprocessing.")


    # --- Post-processing function (Adapted from previous script) ---
    def postprocess_qa_predictions(
        examples_features_map: Dict[str, Any],
        predictions: Tuple[np.ndarray, np.ndarray],
        # ... (other args: version_2_with_negative, n_best_size, etc. - same as before)
        version_2_with_negative: bool = True,
        n_best_size: int = 20,
        max_answer_length: int = 30,
        null_score_diff_threshold: float = 0.0,
        output_dir: Optional[str] = None, # For saving intermediate files
        prefix: Optional[str] = "eval",
        is_world_process_zero: bool = True,
        output_prediction_file: Optional[str] = None,
        output_nbest_file: Optional[str] = None,

    ):
        # ... (rest of the postprocessing function - unchanged from previous version) ...
        # ... (it relies on offset_mapping being present in the feature data) ...
        all_start_logits, all_end_logits = predictions
        all_final_predictions = collections.OrderedDict()
        all_nbest_json_output = collections.OrderedDict()

        logger.info(f"Post-processing {len(examples_features_map)} example predictions...")

        for example_id, data in tqdm(examples_features_map.items()):
            original_context_html = data["context_html"]
            # Clean context once per example for answer extraction
            cleaned_context_for_ans_extraction = clean_html_basic(original_context_html)

            prelim_predictions = []
            min_null_score = None

            for feature in data["features"]:
                global_idx = feature["global_feature_index"]
                start_logits = all_start_logits[global_idx]
                end_logits = all_end_logits[global_idx]
                offset_mapping = feature["offset_mapping"] # Assumes it exists now due to earlier check
                input_ids = feature["input_ids"]

                if not offset_mapping: # Skip if placeholder was added due to missing key earlier
                    logger.warning(f"Skipping feature for example {example_id} due to missing offset_mapping.")
                    continue

                cls_index = input_ids.index(tokenizer.cls_token_id)
                feature_null_score = start_logits[cls_index] + end_logits[cls_index]
                if min_null_score is None or feature_null_score < min_null_score:
                    min_null_score = feature_null_score

                start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
                end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

                for start_index in start_indexes:
                    for end_index in end_indexes:
                        if (start_index >= len(offset_mapping) or end_index >= len(offset_mapping) or
                            offset_mapping[start_index] is None or offset_mapping[end_index] is None or
                            not isinstance(offset_mapping[start_index], (list, tuple)) or len(offset_mapping[start_index]) == 0 or
                            not isinstance(offset_mapping[end_index], (list, tuple)) or len(offset_mapping[end_index]) == 0):
                            continue
                        if offset_mapping[start_index] == (0,0) and offset_mapping[end_index] == (0,0): # Avoid CLS only as answer
                            if start_index == cls_index and end_index == cls_index : continue
                        if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                            continue

                        prelim_predictions.append({
                            "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        })

            if version_2_with_negative:
                if min_null_score is None:
                    min_null_score = -1e9
                prelim_predictions.append({
                    "offsets": (0,0), "score": min_null_score,
                    "start_logit": 0.0, "end_logit": 0.0
                })

            predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

            if version_2_with_negative and not any(p["offsets"] == (0,0) for p in predictions) and min_null_score is not None:
                predictions.append({"offsets": (0,0), "score": min_null_score, "start_logit":0.0, "end_logit":0.0})


            nbest = []
            for pred in predictions:
                offsets = pred.pop("offsets")
                if offsets == (0,0):
                    pred["text"] = ""
                else:
                    pred["text"] = cleaned_context_for_ans_extraction[offsets[0]:offsets[1]].strip()
                nbest.append(pred)

            if not nbest:
                nbest.append({"text": "", "score": 0.0, "start_logit":0.0, "end_logit":0.0})

            # Select best answer
            if version_2_with_negative:
                best_non_null = next((p for p in nbest if p["text"] != ""), None)
                current_null_score = next((p["score"] for p in nbest if p["text"] == ""), -1e9 if min_null_score is None else min_null_score)

                if best_non_null is None or (current_null_score - best_non_null["score"]) > null_score_diff_threshold:
                    all_final_predictions[example_id] = ""
                else:
                    all_final_predictions[example_id] = best_non_null["text"]
            else:
                all_final_predictions[example_id] = nbest[0]["text"]

            all_nbest_json_output[example_id] = [
                {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in p.items()}
                for p in nbest
            ]
        # Save predictions
        if output_prediction_file and is_world_process_zero:
            pred_dir = os.path.dirname(output_prediction_file)
            if pred_dir and not os.path.exists(pred_dir): os.makedirs(pred_dir, exist_ok=True)
            with open(output_prediction_file, "w") as f: json.dump(all_final_predictions, f, indent=4)
        if output_nbest_file and is_world_process_zero:
            nbest_dir = os.path.dirname(output_nbest_file)
            if nbest_dir and not os.path.exists(nbest_dir): os.makedirs(nbest_dir, exist_ok=True)
            with open(output_nbest_file, "w") as f: json.dump(all_nbest_json_output, f, indent=4)

        return all_final_predictions


    # --- Compute Metrics function ---
    metric = evaluate.load("squad_v2")

    def compute_metrics_fn(p: EvalPrediction):
        # Construct output file paths based on output_dir
        prefix = "eval" # Default prefix for eval files
        pred_file = os.path.join(training_args.output_dir, f"{prefix}_predictions.json")
        nbest_file = os.path.join(training_args.output_dir, f"{prefix}_nbest_predictions.json")

        final_predictions = postprocess_qa_predictions(
            examples_features_map=eval_features_for_postprocessing,
            predictions=p.predictions,
            version_2_with_negative=True, # NQ needs this
            n_best_size=data_args.n_best_size,
            max_answer_length=data_args.max_answer_length,
            null_score_diff_threshold=data_args.null_score_diff_threshold,
            output_dir=training_args.output_dir,
            prefix=prefix,
            is_world_process_zero=trainer.is_world_process_zero(),
            output_prediction_file=pred_file,
            output_nbest_file=nbest_file
        )

        formatted_predictions = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in final_predictions.items()]
        references = []
        logger.info("Constructing references for SQuAD v2 metric...")
        for ex_id, data in tqdm(eval_features_for_postprocessing.items()):
            ground_truths_text = []
            annotation = data["annotations"][0] if data.get("annotations") else {}
            short_answers = annotation.get("short_answers", [])
            has_short = False
            if short_answers:
                html_context = data.get("context_html", "")
                for ans_info in short_answers:
                    if ('start_byte' in ans_info and 'end_byte' in ans_info and
                        ans_info['start_byte'] < ans_info['end_byte']):
                        try:
                            ans_bytes = html_context.encode('utf-8')[ans_info['start_byte']:ans_info['end_byte']]
                            ans_text = ans_bytes.decode('utf-8', errors='ignore').strip()
                            if ans_text:
                                ground_truths_text.append(ans_text)
                                has_short = True
                        except Exception: pass

            yes_no_answer = annotation.get("yes_no_answer")
            if yes_no_answer in ["YES", "NO"] and not has_short:
                ground_truths_text.append(yes_no_answer)

            references.append({
                "id": str(ex_id),
                "answers": {"text": ground_truths_text, "answer_start": [0]*len(ground_truths_text)},
            })

        eval_metric_results = metric.compute(predictions=formatted_predictions, references=references)
        logger.info(f"Eval results: {eval_metric_results}")
        return eval_metric_results


    # --- Initialize Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset_features, # Pass the preprocessed features
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics_fn,
    )

    # --- Evaluation ---
    logger.info("*** Evaluating Pre-trained Model ***")
    try:
        metrics = trainer.evaluate() # This will call compute_metrics_fn
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics) # Save metrics to output_dir

        logger.info(f"Predictions saved to: {os.path.join(training_args.output_dir, 'eval_predictions.json')}")
        logger.info(f"N-best predictions saved to: {os.path.join(training_args.output_dir, 'eval_nbest_predictions.json')}")

    except Exception as eval_err:
         logger.error(f"Evaluation failed: {eval_err}", exc_info=True)

    logger.info("Model evaluation script complete.")
    logger.info(f"Results and metrics saved to: {training_args.output_dir}")


if __name__ == "__main__":
    main()
