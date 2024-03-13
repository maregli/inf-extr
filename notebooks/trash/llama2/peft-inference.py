
from peft import PeftModel

import torch

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src import paths
from src.utils import load_ms_data, prepare_ms_data, get_DataLoader, load_model_and_tokenizer, perform_inference, check_gpu_memory

from tqdm import tqdm

import argparse

MODEL_NAME = 'llama2'
QUANTIZATION = "4bit"

BATCH_SIZE = 4

DATA = "original"

SPLIT = "test"

NUM_LABELS = 3

def parse_args():
    parser = argparse.ArgumentParser(description="Zero Shot Classification with Llama2-Chat")
    parser.add_argument("--job_id", type=str, default="unknown", help="Job ID")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME, help="Name of model to be used. Defaults to llama2. Must be saved in the path: paths.MODEL_PATH/model_name")
    parser.add_argument("--peft_model_names", nargs="+", type=str, help="List of PEFT models for which to perform inference. Must be saved in the path: paths.MODEL_PATH/model_name. Must be compatible with base model.")
    parser.add_argument("--quantization", type=str, default=QUANTIZATION, help="Quantization. Must be one of 4bit, bfloat16 or float16. Defaults to 4bit")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch Size. Defaults to 4")
    parser.add_argument("--split", type=str, default=SPLIT, help="Split. Must be one of train, validation or test. Defaults to train")
    parser.add_argument("--num_labels", type=int, default=NUM_LABELS, help="Number of labels (classes) to predict. Defaults to 3. Must be compatible with base model.")

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
    PEFT_MODEL_NAMES = args.peft_model_names
    QUANTIZATION = args.quantization
    BATCH_SIZE = args.batch_size
    SPLIT = args.split
    NUM_LABELS = args.num_labels
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Check GPU Memory
    check_gpu_memory()

    # Load Model and Tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name=MODEL_NAME, quantization=QUANTIZATION, num_labels=NUM_LABELS)

    print("Loaded Model and Tokenizer")

    # Inference
    for peft_model_name in PEFT_MODEL_NAMES:

        print(f"Starting Inference for PEFT Model: {peft_model_name}")

        # Load PEFT Model
        peft_model = PeftModel.from_pretrained(model, paths.MODEL_PATH/peft_model_name)
        print(peft_model.peft_config)

        # Load Data in format matching the PEFT Model configuration
        df = load_ms_data(data=DATA)

        # Prepare Data
        truncation_size = int(peft_model_name.split("_")[-1])
        peft_type = peft_model_name.split("_")[-3]

        if peft_type in ["prompt", "ptune"]:
            is_prompt_tuning = True
            num_virtual_tokens = peft_model.peft_config["default"].num_virtual_tokens
        else:
            is_prompt_tuning = False
            num_virtual_tokens = 0
        
        encoded_dataset = prepare_ms_data(df, tokenizer, is_prompt_tuning = is_prompt_tuning, num_virtual_tokens = num_virtual_tokens, truncation_size=truncation_size)

        # Get DataLoaders
        dataloader = get_DataLoader(encoded_dataset[SPLIT], tokenizer, batch_size=BATCH_SIZE, shuffle=False)

        # Perform Inference
        inference_results = perform_inference(peft_model, dataloader, device)

        # Save Inference Results
        torch.save(inference_results, paths.RESULTS_PATH/"ms-diag"/f"{peft_model_name}.pt")

    return

if __name__ == "__main__":
    main()

