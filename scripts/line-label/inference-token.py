
from peft import PeftModel

import torch

from transformers import Trainer, TrainingArguments, DataCollatorForTokenClassification

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src import paths
from src.utils import (load_line_label_token_data, 
                       load_model_and_tokenizer,
                       tokenize_and_align_labels, 
                       check_gpu_memory, 
                       get_results_from_token_preds,
                       line_label_token_id2label,
                       )

from tqdm import tqdm

import argparse

import evaluate

import numpy as np

import os

def parse_args():
    parser = argparse.ArgumentParser(description="Inference for Line Label Token Classification Task")
    parser.add_argument("--job_id", type=str, default="unknown", help="Job ID")
    parser.add_argument("--model_name", type=str, default="medbert-512", help="Name of model to be used. Defaults to medbert-512. Must be saved in the path: paths.MODEL_PATH/model_name")
    parser.add_argument("--peft_model_name", type=str, help="PEFT model for which to perform inference. Must be saved in the path: paths.MODEL_PATH/model_name. Must be compatible with base model.")
    parser.add_argument("--quantization", type=str, default=None, help="Quantization. Must be one of 4bit, bfloat16 or float16. Defaults to None")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch Size. Defaults to 4")
    parser.add_argument("--split", type=str, default="test", help="Split. Must be one of train, validation or test. Defaults to test")
    parser.add_argument("--task_type", type=str, default="token", help="Task Type. Must be one of class, token or clm. Defaults to token")
    parser.add_argument("--output_hidden_states", action="store_true", help="Whether to output hidden states. Defaults to False")


    args = parser.parse_args()

    # Print or log the parsed arguments
    print("Parsed Arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    return args


seqeval = evaluate.load(os.path.join(paths.METRICS_PATH,"seqeval"))
label_list = list(line_label_token_id2label.values())
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def main():

    args = parse_args()
    JOB_ID = args.job_id
    MODEL_NAME = args.model_name
    PEFT_MODEL_NAME = args.peft_model_name
    QUANTIZATION = args.quantization
    BATCH_SIZE = args.batch_size
    SPLIT = args.split
    TASK_TYPE = args.task_type
    OUTPUT_HIDDEN_STATES = args.output_hidden_states


    
    # Check GPU Memory
    check_gpu_memory()

    # Load data
    dataset_token = load_line_label_token_data()

    # Load Model and Tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name=MODEL_NAME,
                                                task_type=TASK_TYPE,
                                                quantization=QUANTIZATION,
                                                )
    
    if PEFT_MODEL_NAME:
        model = PeftModel.from_pretrained(model, paths.MODEL_PATH/PEFT_MODEL_NAME)

    print("Loaded Model and Tokenizer")

    # Prepare Data
    encoded_dataset = dataset_token.map(tokenize_and_align_labels, batched=True, fn_kwargs={"tokenizer": tokenizer})

    # Trainer will automaticall do all padding and dataloading stuff
    training_args = TrainingArguments(
        per_device_eval_batch_size=BATCH_SIZE,
        output_dir=paths.RESULTS_PATH/"line_label"/f"{MODEL_NAME}",
        )
    
    trainer = Trainer(model=model,
                    data_collator=DataCollatorForTokenClassification(tokenizer),
                    args=training_args,
                    compute_metrics=compute_metrics,
                    )

    # Perform Inference
    predictions, labels, metrics = trainer.predict(encoded_dataset[SPLIT])

    # Get Results
    results = get_results_from_token_preds(predictions=predictions, dataset=encoded_dataset, tokenizer=tokenizer, split=SPLIT)
    
    if OUTPUT_HIDDEN_STATES:
        print("Outputting Hidden States")
        labels = []
        last_hidden_states = []
        for report_idx in tqdm(range(len(encoded_dataset[SPLIT]))):
            # Input ids, attention mask to tensor
            input_ids = torch.tensor(encoded_dataset["test"]["input_ids"][report_idx]).unsqueeze(0).to(model.device)
            attention_mask = torch.tensor(encoded_dataset["test"]["attention_mask"][report_idx]).unsqueeze(0).to(model.device)

            # Get hidden states
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                
            last_hidden_state = outputs.hidden_states[-1].to("cpu")

            last_hidden_states.append(last_hidden_state.squeeze(0))
            labels.extend(encoded_dataset[SPLIT]["labels"][report_idx])

        last_hidden_states = torch.cat(last_hidden_states, dim = 0)

        results_hidden_states = {
            "last_hidden_states": last_hidden_states,
            "labels": labels,
        }

    saving_model_name = PEFT_MODEL_NAME if PEFT_MODEL_NAME else MODEL_NAME

    # Save Inference Results
    print("Saving Inference Results at:", paths.RESULTS_PATH/"line-label"/f"{saving_model_name}_{SPLIT}.pt")
    torch.save(results, paths.RESULTS_PATH/"line-label"/f"{saving_model_name}_{SPLIT}.pt")
    print("Saving Last Hidden States at:", paths.RESULTS_PATH/"line-label"/f"{saving_model_name}_{SPLIT}_hidden_states.pt")
    torch.save(results_hidden_states, paths.RESULTS_PATH/"line-label"/f"{saving_model_name}_{SPLIT}_hidden_states.pt")

    return

if __name__ == "__main__":
    main()


