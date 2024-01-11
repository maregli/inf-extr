
import torch

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src import paths
from src.utils import load_model_and_tokenizer, load_ms_data, prepare_ms_data, get_DataLoader, check_gpu_memory, train_loop

import argparse

MODEL_NAME = 'medbert'
QUANTIZATION = "4bit"

TRUNCATION_SIZE = 256

BATCH_SIZE = 4
LEARNING_RATE = 1e-3
NUM_EPOCHS = 4

GRADIENT_ACCUMULATION_STEPS = None

PEFT_TYPE = "lora"

DATA = "original"

NUM_LABELS = 3



def parse_args():
    parser = argparse.ArgumentParser(description="Zero Shot Classification with Llama2-Chat")
    parser.add_argument("--job_id", type=str, default="unknown", help="Job ID")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME, help="Name of base model to be used. Defaults to medbert. Must be saved in the path: paths.MODEL_PATH/model_name")
    parser.add_argument("--quantization", type=str, default=QUANTIZATION, help="Quantization. Must be one of 4bit, bfloat16 or float16. Defaults to 4bit")
    parser.add_argument("--truncation_size", type=int, default=TRUNCATION_SIZE, help="Truncation Size of the input text. Defaults to 256")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch Size. Defaults to 4")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning Rate. Defaults to 1e-3")
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS, help="Number of Epochs. Defaults to 4")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=GRADIENT_ACCUMULATION_STEPS, help="Gradient Accumulation Steps. If not set, it will be set to 8 // batch_size or 1 if batch_size >= 8")
    parser.add_argument("--data", type=str, default=DATA, help="Data. Must be one of original, zero-shot or augmented. Defaults to original")
    parser.add_argument("--num_labels", type=int, default=NUM_LABELS, help="Number of Labels. Defaults to 3")

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
    QUANTIZATION = args.quantization
    TRUNCATION_SIZE = args.truncation_size
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    NUM_EPOCHS = args.num_epochs
    GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation_steps
    DATA = args.data
    NUM_LABELS = args.num_labels

    if GRADIENT_ACCUMULATION_STEPS is None:
        if BATCH_SIZE >= 8:
            GRADIENT_ACCUMULATION_STEPS = 1
        else:
            GRADIENT_ACCUMULATION_STEPS = 8 // BATCH_SIZE

    # Name Model for saving
    finetuned_model_name = f"ms-diag_{MODEL_NAME}_{QUANTIZATION}_finetuned_{DATA}_{TRUNCATION_SIZE}"
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Check GPU Memory
    check_gpu_memory()

    # Load Model and Tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name=MODEL_NAME, quantization=QUANTIZATION, num_labels=NUM_LABELS)

    print("Loaded Model and Tokenizer")

    # Load Data
    df = load_ms_data(data=DATA)

    # Prepare Data
    encoded_dataset = prepare_ms_data(df, tokenizer, truncation_size=TRUNCATION_SIZE)

    print("Loaded Data")

    # Get DataLoaders
    train_dataloader = get_DataLoader(encoded_dataset["train"], tokenizer, batch_size=BATCH_SIZE)
    eval_dataloader = get_DataLoader(encoded_dataset["validation"], tokenizer, batch_size=BATCH_SIZE, shuffle=False)

    # Train Loop
    print("Starting Training")
    train_loop(model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                device=device,
                num_epochs=NUM_EPOCHS,
                learning_rate=LEARNING_RATE,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                finetuned_model_name=finetuned_model_name,
                )
    print("Training Finished")

    # Save Tokenizer
    print("Saving Tokenizer at:", paths.MODEL_PATH/finetuned_model_name)
    tokenizer.save_pretrained(paths.MODEL_PATH/finetuned_model_name)
    return

if __name__ == "__main__":
    main()


