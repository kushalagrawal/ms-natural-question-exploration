import os
import logging
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
import collections

import numpy as np
import torch # Ensure torch is installed for transformers
import datasets
# <<< Updated Import: Use evaluate instead of datasets for metrics >>>
import evaluate
from datasets import load_from_disk # Keep this for loading data
# <<< End Update >>>
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
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
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
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
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    processed_dataset_dir: str = field(
        metadata={"help": "Path to the directory containing the preprocessed Arrow datasets (train and dev splits)."}
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    n_best_size: int = field(
        default=20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    null_score_diff_threshold: float = field(
        default=0.0, # SQuAD v2.0 default. Needs tuning for NQ.
        metadata={
            "help": "The threshold used to select the null answer: if the best answer has a score that is less than "
            "the score of the null answer minus this threshold, the null answer is selected for SQuAD v2.0."
            "A A lower threshold makes it more likely to predict a non-null answer."
        },
    )
    output_prediction_file: Optional[str] = field(
        default=None, metadata={"help": "If provided, the path to save predictions."}
    )
    output_nbest_file: Optional[str] = field(
        default=None, metadata={"help": "If provided, the path to save nbest predictions."}
    )
    # <<< NEW: Argument to load original data for evaluation context/references >>>
    original_dev_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the original NQ dev .jsonl file (required for accurate evaluation)."}
    )


# --- Main Script ---
def main():
    # Parse arguments
    # In a real script, you'd use HfArgumentParser or argparse
    # For this example, we'll define them directly for clarity
    # You would replace these with command-line arguments or a config file
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    # Example command line:
    # python your_script.py --model_name_or_path bert-base-uncased --processed_dataset_dir ./nq_processed_final --original_dev_path ./unarchived/v1.0/dev/nq-dev-00.jsonl --output_dir ./results_bert --do_train --do_eval --num_train_epochs 1 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --learning_rate 3e-5 --max_seq_length 384 --doc_stride 128 --save_strategy no --evaluation_strategy steps --eval_steps 500 --logging_steps 100
    # For this run, we define them directly:
    model_args = ModelArguments(
        model_name_or_path="bert-base-uncased" # Example: Change to "roberta-base", etc.
    )
    data_args = DataTrainingArguments(
        processed_dataset_dir="./nq_processed_final", # Path from preprocessing
        max_seq_length=384,
        doc_stride=128,
        null_score_diff_threshold = 0.0, # Tune this!
        # <<< *** IMPORTANT: Set this path correctly for evaluation *** >>>
        original_dev_path="./unarchived/v1.0/dev/nq-dev-00.jsonl" # Example path, adjust as needed. Might need to combine if multiple dev files.
    )
    training_args = TrainingArguments(
        output_dir=f"./results_{model_args.model_name_or_path.replace('/', '_')}", # Dir to save model
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=8, # Adjust based on GPU memory
        per_device_eval_batch_size=16, # Can often use larger eval batch size
        learning_rate=3e-5,
        num_train_epochs=1, # Start with 1 epoch for NQ, can increase if needed
        warmup_ratio=0.1, # Use ratio instead of steps for variable dataset sizes
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100, # Log more frequently
        # <<< UPDATED PARAMETER NAMES >>>
        eval_strategy="epoch",   # Changed from evaluation_strategy
        save_strategy="epoch",   # Changed from save_strategy
        # <<< END UPDATE >>>
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none"
        # fp16=True, # Enable mixed precision if GPU supports it & helps
    )

    # --- Argument Validation ---
    if training_args.do_eval and not data_args.original_dev_path:
        logger.error("Evaluation requires --original_dev_path to load ground truth answers and contexts.")
        return
    if training_args.do_eval and not os.path.exists(data_args.original_dev_path):
         logger.error(f"Original dev file not found at: {data_args.original_dev_path}")
         return


    logger.info("-------------------- Configuration --------------------")
    logger.info(f"Model Arguments: {model_args}")
    logger.info(f"Data Arguments: {data_args}")
    logger.info(f"Training Arguments: {training_args}")
    logger.info("-----------------------------------------------------")


    # --- Load Tokenizer and Model ---
    logger.info(f"Loading tokenizer: {model_args.tokenizer_name or model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
    )
    # Ensure pad token is set for padding (needed by DataCollator)
    if tokenizer.pad_token is None:
        logger.warning("Tokenizer does not have a pad token, setting it to eos_token for padding.")
        tokenizer.pad_token = tokenizer.eos_token # Common practice if pad_token is missing


    logger.info(f"Loading model: {model_args.model_name_or_path}")
    model_config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        config=model_config,
        cache_dir=model_args.cache_dir,
    )
    # If tokenizer required adding pad token, resize model embeddings
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
         model.resize_token_embeddings(len(tokenizer))


    # --- Load Preprocessed Datasets ---
    logger.info(f"Loading preprocessed datasets from: {data_args.processed_dataset_dir}")
    try:
        train_dataset_path = os.path.join(data_args.processed_dataset_dir, "train")
        eval_dataset_path = os.path.join(data_args.processed_dataset_dir, "dev")

        if training_args.do_train and not os.path.exists(train_dataset_path):
            logger.error(f"Training dataset not found at {train_dataset_path}")
            return
        if training_args.do_eval and not os.path.exists(eval_dataset_path):
            logger.error(f"Evaluation dataset not found at {eval_dataset_path}")
            return

        raw_datasets = datasets.DatasetDict()
        if training_args.do_train:
            raw_datasets["train"] = load_from_disk(train_dataset_path)
        if training_args.do_eval:
            raw_datasets["validation"] = load_from_disk(eval_dataset_path)
            # Ensure validation set has offset_mapping if needed by postprocessor
            # Our preprocessing script saves it, but Trainer might remove it.
            # Keep columns needed for postprocessing
            validation_features = raw_datasets["validation"]
            # Add offset_mapping back if it was removed (Trainer removes unused columns)
            # This requires the preprocessing script to have saved it.
            # Let's assume it's there for now. If errors occur, modify preprocessing.

        logger.info(f"Datasets loaded: {raw_datasets}")

    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        return

    # --- Load Original Validation Data (for context and answers) ---
    original_validation_data = {}
    if training_args.do_eval:
        logger.info(f"Loading original validation data from: {data_args.original_dev_path}")
        try:
            # <<< Handle multiple dev files if necessary >>>
            if os.path.isdir(data_args.original_dev_path):
                 original_files = [os.path.join(data_args.original_dev_path, f)
                                   for f in os.listdir(data_args.original_dev_path) if f.endswith(".jsonl")]
            elif os.path.isfile(data_args.original_dev_path):
                 original_files = [data_args.original_dev_path]
            else:
                 logger.error(f"original_dev_path is neither a file nor a directory: {data_args.original_dev_path}")
                 return

            for filepath in original_files:
                logger.info(f"Loading original examples from: {filepath}")
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        example = json.loads(line)
                        example_id = str(example.get("example_id")) # Ensure string ID
                        if example_id:
                            original_validation_data[example_id] = {
                                "context": example.get("document_html", ""), # Store original HTML
                                "annotations": example.get("annotations", [])
                            }
            logger.info(f"Loaded context/annotations for {len(original_validation_data)} original validation examples.")
        except Exception as e:
            logger.error(f"Error loading original validation data: {e}")
            return


    # --- Post-processing function ---
    # Needs access to original context and preprocessed features' offset mapping
    def postprocess_qa_predictions(
        examples_features_map: Dict[str, Any], # Combine original examples and features
        predictions: Tuple[np.ndarray, np.ndarray],
        version_2_with_negative: bool = True,
        n_best_size: int = 20,
        max_answer_length: int = 30,
        null_score_diff_threshold: float = 0.0,
        output_dir: Optional[str] = None,
        prefix: Optional[str] = None,
        log_level: Optional[int] = logging.WARNING,
    ):
        # examples_features_map should map example_id -> {"context": str, "features": [feature_dict]}
        # where feature_dict contains offset_mapping, input_ids, etc.

        if len(predictions) != 2:
            raise ValueError("`predictions` should be a tuple with two elements (start_logits, end_logits).")
        all_start_logits, all_end_logits = predictions

        # The predictions array corresponds flattened features, need to map back
        # We need a flat list of features with their example_id and offset_mapping
        flat_features = []
        for ex_id, data in examples_features_map.items():
            for feature in data["features"]:
                flat_features.append(feature) # feature already has example_id

        if len(all_start_logits) != len(flat_features):
             raise ValueError(f"Got {len(all_start_logits)} predictions and {len(flat_features)} features.")

        # The dictionaries to store our final predictions.
        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()

        logger.info(f"Post-processing {len(examples_features_map)} example predictions split into {len(flat_features)} features.")

        # Iterate through original examples
        for example_id, example_data in tqdm(examples_features_map.items()):
            min_null_prediction = None
            prelim_predictions = []
            context = example_data["context"] # Get original HTML context

            # Iterate through features associated with this example
            for feature_index, feature in enumerate(example_data["features"]):
                # Find the prediction index corresponding to this feature
                try:
                    # Use the global index stored during mapping
                    global_feature_index = feature["global_feature_index"]
                    pred_start_logits = all_start_logits[global_feature_index]
                    pred_end_logits = all_end_logits[global_feature_index]
                    offset_mapping = feature["offset_mapping"]
                    token_type_ids = feature.get("token_type_ids")

                    # Calculate null score for this feature
                    # Ensure input_ids is present in feature data
                    input_ids = feature["input_ids"]
                    cls_index = input_ids.index(tokenizer.cls_token_id)
                    feature_null_score = pred_start_logits[cls_index] + pred_end_logits[cls_index]
                    if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
                        min_null_prediction = {"score": feature_null_score} # Store minimal info needed

                    # Find potential answer spans
                    start_indexes = np.argsort(pred_start_logits)[-1 : -n_best_size - 1 : -1].tolist()
                    end_indexes = np.argsort(pred_end_logits)[-1 : -n_best_size - 1 : -1].tolist()
                    for start_index in start_indexes:
                        for end_index in end_indexes:
                            if (start_index >= len(offset_mapping) or end_index >= len(offset_mapping) or
                                offset_mapping[start_index] is None or offset_mapping[end_index] is None or
                                not isinstance(offset_mapping[start_index], (list, tuple)) or
                                not isinstance(offset_mapping[end_index], (list, tuple)) or
                                len(offset_mapping[start_index]) == 0 or len(offset_mapping[end_index]) == 0 ):
                                continue
                            if offset_mapping[start_index] == (0, 0) or offset_mapping[end_index] == (0, 0):
                                continue
                            if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                                continue

                            start_char = offset_mapping[start_index][0]
                            end_char = offset_mapping[end_index][1]
                            prelim_predictions.append({
                                "start": start_char, "end": end_char,
                                "score": pred_start_logits[start_index] + pred_end_logits[end_index],
                                "start_logit": pred_start_logits[start_index],
                                "end_logit": pred_end_logits[end_index],
                            })
                except Exception as e:
                     logger.warning(f"Error processing feature {feature.get('global_feature_index', 'N/A')} for example {example_id}: {e}")
                     continue # Skip this feature on error


            if version_2_with_negative and min_null_prediction is not None:
                # Add a dummy prediction for the null answer score comparison
                prelim_predictions.append({
                    "start": 0, "end": 0, # Represents null answer
                    "score": min_null_prediction["score"],
                    "start_logit": 0, "end_logit": 0 # Logits not needed for comparison logic here
                })
                null_score = min_null_prediction["score"]

            # Get the best predictions
            predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

            # If null answer wasn't included in top N, add it back for comparison
            has_null = any(p["start"] == 0 and p["end"] == 0 for p in predictions)
            if version_2_with_negative and min_null_prediction is not None and not has_null:
                 predictions.append({ "start": 0, "end": 0, "score": null_score, "start_logit":0, "end_logit":0})

            # Map offsets to text
            nbest = []
            for pred in predictions:
                if pred["start"] == 0 and pred["end"] == 0: # Handle null prediction
                     nbest.append({
                         "text": "",
                         "score": pred["score"],
                         "start_logit": pred["start_logit"],
                         "end_logit": pred["end_logit"],
                     })
                else:
                    start_char, end_char = pred["start"], pred["end"]
                    # Extract text from original HTML context
                    answer_text = context[start_char:end_char].strip()

                    nbest.append({
                        "text": answer_text,
                        "score": pred["score"],
                        "start_logit": pred["start_logit"],
                        "end_logit": pred["end_logit"],
                    })

            # Handle cases where no valid predictions are found
            if not nbest:
                nbest.append({"text": "", "score": 0.0, "start_logit": 0.0, "end_logit": 0.0})

            # Select best answer based on threshold
            best_non_null = next((p for p in nbest if p["text"] != ""), None)
            if version_2_with_negative and min_null_prediction is not None:
                # Use the score of the best non-null prediction vs null score
                if best_non_null is not None:
                     score_diff = null_score - best_non_null["score"]
                     if score_diff > null_score_diff_threshold:
                         all_predictions[example_id] = "" # Predict no answer
                     else:
                         all_predictions[example_id] = best_non_null["text"]
                else:
                     # Only null prediction was possible
                     all_predictions[example_id] = ""
            elif best_non_null is not None:
                 all_predictions[example_id] = best_non_null["text"]
            else: # Only null answer was generated even without v2 logic
                 all_predictions[example_id] = ""


            # Format n-best list for saving
            all_nbest_json[example_id] = [
                {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in p.items()}
                for p in nbest
            ]

        # Save predictions if path is provided
        # <<< Check if output_dir exists >>>
        is_world_process_zero = trainer.is_world_process_zero() if 'trainer' in globals() else True # Simple check if trainer exists
        if output_prediction_file is not None and is_world_process_zero:
             output_pred_dir = os.path.dirname(output_prediction_file)
             if output_pred_dir and not os.path.exists(output_pred_dir): os.makedirs(output_pred_dir, exist_ok=True)
             with open(output_prediction_file, "w") as writer:
                 writer.write(json.dumps(all_predictions, indent=4) + "\n")
        if output_nbest_file is not None and is_world_process_zero:
             output_nbest_dir = os.path.dirname(output_nbest_file)
             if output_nbest_dir and not os.path.exists(output_nbest_dir): os.makedirs(output_nbest_dir, exist_ok=True)
             with open(output_nbest_file, "w") as writer:
                 writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

        return all_predictions


    # --- Compute Metrics function ---
    # <<< Updated Metric Loading >>>
    metric = evaluate.load("squad_v2")
    # <<< End Update >>>

    # --- Prepare data needed for postprocessing during evaluation ---
    # We need a map from example_id to its original context and its features
    # The features list should contain the offset_mapping and input_ids
    eval_features_for_postprocessing = collections.defaultdict(lambda: {"features": []})

    if training_args.do_eval:
        # Add a global index to each feature in the validation set
        # Use remove_columns to avoid keeping the original large dataset in memory if not needed elsewhere
        validation_features_indexed = raw_datasets["validation"].add_column(
            "global_feature_index", range(len(raw_datasets["validation"]))
        )
        # Select only columns needed for postprocessing mapping + the index
        columns_to_keep = ["example_id", "offset_mapping", "input_ids", "token_type_ids", "global_feature_index"]
        # Filter out token_type_ids if not present
        columns_to_keep = [col for col in columns_to_keep if col in validation_features_indexed.column_names]

        validation_features_mapped = validation_features_indexed.map(
            lambda x: x, # Identity function, just to select columns
            batched=True,
            remove_columns=[col for col in validation_features_indexed.column_names if col not in columns_to_keep]
        )


        # Populate the map
        logger.info("Mapping evaluation features to original examples...")
        for feature in tqdm(validation_features_mapped): # Use the mapped dataset
            example_id = feature["example_id"]
            if example_id not in eval_features_for_postprocessing:
                # Fetch original context if not already stored
                if example_id in original_validation_data:
                    # Store original HTML context
                    eval_features_for_postprocessing[example_id]["context"] = original_validation_data[example_id]["context"]
                    # Store original annotations for reference construction
                    eval_features_for_postprocessing[example_id]["annotations"] = original_validation_data[example_id]["annotations"]
                else:
                    logger.warning(f"Could not find original context/annotations for example_id: {example_id}")
                    eval_features_for_postprocessing[example_id]["context"] = "" # Add empty context
                    eval_features_for_postprocessing[example_id]["annotations"] = []

            # Add feature details needed for postprocessing
            # Feature already contains the necessary keys after the map operation
            eval_features_for_postprocessing[example_id]["features"].append(feature)
        logger.info("Finished mapping features.")


    def compute_metrics_fn(p: EvalPrediction):
        # p.predictions contains the raw logits (start_logits, end_logits)
        # p.label_ids contains the ground truth start/end positions (not directly used by postprocessor)
        # The order of predictions corresponds to the order of the eval_dataset Features

        # Run postprocessing
        # Construct the output file paths dynamically if needed
        pred_file = os.path.join(training_args.output_dir, "eval_predictions.json") if data_args.output_prediction_file is None else data_args.output_prediction_file
        nbest_file = os.path.join(training_args.output_dir, "eval_nbest_predictions.json") if data_args.output_nbest_file is None else data_args.output_nbest_file

        final_predictions = postprocess_qa_predictions(
            examples_features_map=eval_features_for_postprocessing, # Pass the map
            predictions=p.predictions,
            version_2_with_negative=True,
            n_best_size=data_args.n_best_size,
            max_answer_length=data_args.max_answer_length,
            null_score_diff_threshold=data_args.null_score_diff_threshold,
            output_dir=training_args.output_dir,
            prefix="eval",
            # Pass trainer reference if available, otherwise assume main process
            is_world_process_zero=getattr(trainer, 'is_world_process_zero', lambda: True)(),
            output_prediction_file=pred_file, # Pass path to save predictions
            output_nbest_file=nbest_file # Pass path to save nbest
        )

        # Format predictions and references for SQuAD v2 metric
        formatted_predictions = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in final_predictions.items()]

        # --- Construct References from Original Data ---
        references = []
        logger.info("Constructing references for metric computation...")
        # Use the keys from the map which represent unique example_ids evaluated
        for ex_id in tqdm(eval_features_for_postprocessing.keys()):
            data = eval_features_for_postprocessing[ex_id]
            ground_truths = []
            answer_starts = []
            has_short_answer = False
            # Use the stored original annotations
            if data.get("annotations"):
                annotation = data["annotations"][0] # Use first annotation
                short_answers = annotation.get("short_answers", [])
                if short_answers:
                    # Extract text for all valid short answers
                    html_context = data.get("context", "")
                    for ans in short_answers:
                         if ('start_byte' in ans and 'end_byte' in ans and
                             isinstance(ans['start_byte'], int) and isinstance(ans['end_byte'], int) and
                             ans['start_byte'] >= 0 and ans['end_byte'] >= 0 and
                             ans['start_byte'] < ans['end_byte']):
                             try:
                                 ans_bytes = html_context.encode('utf-8')[ans['start_byte']:ans['end_byte']]
                                 ans_text = ans_bytes.decode('utf-8', errors='ignore').strip()
                                 if ans_text: # Only add non-empty answers
                                      ground_truths.append(ans_text)
                                      answer_starts.append(0) # Placeholder start offset
                                      has_short_answer = True
                             except Exception:
                                 pass # Ignore errors extracting specific answers

                # Handle Yes/No
                yes_no_answer = annotation.get("yes_no_answer")
                if yes_no_answer in ["YES", "NO"] and not has_short_answer:
                     ground_truths.append(yes_no_answer)
                     answer_starts.append(0) # Placeholder start offset

            # Format for SQuAD metric
            references.append({
                "id": str(ex_id),
                "answers": {"text": ground_truths, "answer_start": answer_starts},
            })
        logger.info("Finished constructing references.")

        # Compute the metric
        eval_metric_results = metric.compute(predictions=formatted_predictions, references=references)
        logger.info(f"Eval results: {eval_metric_results}")
        return eval_metric_results


    # --- Initialize Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=raw_datasets.get("train"), # Use .get() for safety
        # Pass the features dataset for evaluation, compute_metrics will use the mapped data
        eval_dataset=raw_datasets.get("validation"),
        tokenizer=tokenizer,
        data_collator=default_data_collator, # Handles padding
        compute_metrics=compute_metrics_fn if training_args.do_eval else None,
    )

    # --- Training ---
    if training_args.do_train:
        logger.info("*** Training ***")
        try:
            # Use resume_from_checkpoint=True if you want to resume training
            train_result = trainer.train(resume_from_checkpoint=None)
            metrics = train_result.metrics
            trainer.save_model()  # Saves the tokenizer too for fast tokenizers.
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()
        except Exception as train_err:
             logger.error(f"Training failed: {train_err}", exc_info=True)

    # --- Evaluation ---
    if training_args.do_eval:
        logger.info("*** Evaluating ***")
        try:
            metrics = trainer.evaluate() # compute_metrics_fn is called here
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
        except Exception as eval_err:
             logger.error(f"Evaluation failed: {eval_err}", exc_info=True)


    logger.info("Fine-tuning and evaluation script complete.")
    logger.info(f"Results saved to: {training_args.output_dir}")


if __name__ == "__main__":
    main()

