
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src import paths
from src.utils import (load_model_and_tokenizer, 
                       load_line_label_data, 
                       prepare_line_label_data, 
                       check_gpu_memory, 
                       get_optimizer_and_scheduler, 
                       compute_metrics,
)

import argparse

from transformers import Trainer, TrainingArguments, DataCollatorWithPadding

import json

from peft import get_peft_config, get_peft_model


def parse_args():
    parser = argparse.ArgumentParser(description="Zero Shot Classification with Llama2-Chat")
    parser.add_argument("--job_id", type=str, default="unknown", help="Job ID")
    parser.add_argument("--model_name", type=str, default="medbert-512", help="Name of base model to be used. Defaults to medbert-512. Must be saved in the path: paths.MODEL_PATH/model_name")
    parser.add_argument("--peft_config", type=str, default=None, help="PEFT Config. JSON-formatted configuration. Defaults to None")
    parser.add_argument("--quantization", type=str, default=None, help="Quantization. Must be one of 4bit, bfloat16 or float16. Defaults to None")
    parser.add_argument("--task_type", type=str, default="class", help="Task Type. Must be one of class or clm. Defaults to class")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch Size. Defaults to 4")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate. Defaults to 1e-3")
    parser.add_argument("--num_epochs", type=int, default=4, help="Number of Epochs. Defaults to 4")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient Accumulation Steps. Defaults to 1")
    parser.add_argument("--num_labels", type=int, default=None, help="Number of Labels. If not set and task_type is class, it will be automatically inferred from data[\"train\"][\"labels\"]. Defaults to None")

    args = parser.parse_args()

    # Print or log the parsed arguments
    print("Parsed Arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    return args


def main():

    args = parse_args()
    JOB_ID = args.job_id
    MODEL_NAME = args.model_name
    PEFT_CONFIG = args.peft_config
    QUANTIZATION = args.quantization
    TASK_TYPE = args.task_type
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    NUM_EPOCHS = args.num_epochs
    GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation_steps
    NUM_LABELS = args.num_labels

    # Check GPU Memory
    check_gpu_memory()

    # Load Data
    df = load_line_label_data()
    
    if NUM_LABELS is None and TASK_TYPE == "class":
        NUM_LABELS = len(set(df['train']["labels"]))
        print("Automatically inferred NUM_LABELS to:", NUM_LABELS)
    print("Loaded Data")

    # Load Model and Tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name=MODEL_NAME, num_labels=NUM_LABELS, quantization=QUANTIZATION, task_type=TASK_TYPE)

    # If Peft
    # Get PEFT Config
    if PEFT_CONFIG is not None:
        config = json.loads(PEFT_CONFIG)
        PEFT_CONFIG = get_peft_config(config)
        print("PEFT Config:", PEFT_CONFIG)
        model = get_peft_model(model, PEFT_CONFIG)
    else:
        config = {}

    print("Loaded Model and Tokenizer")

    # Prepare Data
    encoded_dataset = prepare_line_label_data(df, tokenizer, inference_mode=False)

    # Num of training steps
    num_training_steps = len(encoded_dataset["train"]) // BATCH_SIZE * NUM_EPOCHS

    # Get optimizer and scheduler
    optimizer, lr_scheduler = get_optimizer_and_scheduler(model=model, num_training_steps=num_training_steps, learning_rate=LEARNING_RATE)

    print("Got Optimizer and Scheduler")

    arg_names = [MODEL_NAME, QUANTIZATION, config.get("peft_type", None), TASK_TYPE]
    arg_names = [arg_name for arg_name in arg_names if arg_name is not None]

    finetuned_model_name = f"line-label_{'_'.join(arg_names)}"

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(paths.MODEL_PATH, "training", finetuned_model_name),          
        num_train_epochs=NUM_EPOCHS,             
        per_device_train_batch_size=BATCH_SIZE,  
        per_device_eval_batch_size=BATCH_SIZE,   
        warmup_steps=0,                               
        logging_dir='./logs',            
        logging_steps=20,
        save_strategy='epoch',
        evaluation_strategy='epoch',
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False},
        seed=42,
        load_best_model_at_end=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        optimizers=(optimizer, lr_scheduler),                        
        args=training_args,                  
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["val"],
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer, padding="longest", pad_to_multiple_of=8),
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


