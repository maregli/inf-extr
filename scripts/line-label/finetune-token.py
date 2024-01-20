
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src import paths
from src.utils import (load_model_and_tokenizer, 
                       load_line_label_token_data, 
                       prepare_line_label_data, 
                       check_gpu_memory, 
                       get_optimizer_and_scheduler, 
                       compute_metrics,
                       tokenize_and_align_labels,
                       select_peft_config,
                        line_label_token_id2label,

)

from peft import get_peft_config, get_peft_model

import argparse

import evaluate

import numpy as np

import json

from transformers import Trainer, TrainingArguments, DataCollatorForTokenClassification


def parse_args():
    parser = argparse.ArgumentParser(description="Line Label Token Finetuning")
    parser.add_argument("--job_id", type=str, default="unknown", help="Job ID")
    parser.add_argument("--model_name", type=str, default="medbert-512", help="Name of base model to be used. Defaults to medbert-512. Must be saved in the path: paths.MODEL_PATH/model_name")
    parser.add_argument("--quantization", type=str, default=None, help="Quantization. Must be one of 4bit, bfloat16 or float16. Defaults to None. If quantization is used PEFT should also be used.")
    parser.add_argument("--task_type", type=str, default="token", help="Task Type. Must be one of class, token or clm. Defaults to class")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch Size. Defaults to 4")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate. Defaults to 1e-3")
    parser.add_argument("--num_epochs", type=int, default=4, help="Number of Epochs. Defaults to 4")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient Accumulation Steps. Defaults to 1")
    parser.add_argument("--peft_config", type=str, default=None, help="PEFT Config. JSON-formatted configuration. Defaults to None")
    parser.add_argument("--attn_implementation", type=str, default=None, help="To implement Flash Attention 2 provide flash_attention_2. Defaults to None.")

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
    QUANTIZATION = args.quantization
    TASK_TYPE = args.task_type
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    NUM_EPOCHS = args.num_epochs
    GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation_steps
    PEFT_CONFIG = args.peft_config
    ATTN_IMPLEMENTATION = args.attn_implementation

    # Check GPU Memory
    check_gpu_memory()

    # Load Data
    dataset_token = load_line_label_token_data()

    print("Loaded Data")

    # Load Model and Tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name=MODEL_NAME, 
                                                task_type=TASK_TYPE, 
                                                quantization=QUANTIZATION, 
                                                attn_implementation=ATTN_IMPLEMENTATION,
                                                )
    
    # Get PEFT Config
    if PEFT_CONFIG is not None:
        config = json.loads(PEFT_CONFIG)
        PEFT_CONFIG = get_peft_config(config)
        model = get_peft_model(model, PEFT_CONFIG)
    else:
        config = {}

    print("Loaded Model and Tokenizer")

    # Prepare Data
    encoded_dataset = dataset_token.map(tokenize_and_align_labels, batched=True, fn_kwargs={"tokenizer": tokenizer})

    # Num of training steps
    num_training_steps = len(encoded_dataset["train"]) // BATCH_SIZE * NUM_EPOCHS

    # Get optimizer and scheduler
    optimizer, lr_scheduler = get_optimizer_and_scheduler(model=model, num_training_steps=num_training_steps, learning_rate=LEARNING_RATE)

    # Get DataCollator
    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)

    print("Got Optimizer, Scheduler and DataCollator")

    arg_names = [MODEL_NAME, QUANTIZATION, config.get("peft_type", None), TASK_TYPE]
    arg_names = [arg_name for arg_name in arg_names if arg_name is not None]

    finetuned_model_name = f"line-label_{'_'.join(arg_names)}"

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(paths.MODEL_PATH, "training", finetuned_model_name),
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False},
        logging_steps=10,
        logging_dir='./logs',
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, lr_scheduler),
    )

    print("Starting Training")

    # Train
    trainer.train()

    print("Finished Training")

    # Save Model
    print("Saving Model at:", paths.MODEL_PATH/finetuned_model_name)
    trainer.save_model(paths.MODEL_PATH/finetuned_model_name)

    # Save Tokenizer
    print("Saving Tokenizer at:", paths.MODEL_PATH/finetuned_model_name)
    tokenizer.save_pretrained(paths.MODEL_PATH/finetuned_model_name)
    return

if __name__ == "__main__":
    main()


