
from peft import PeftModel

import torch

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src import paths
from src.utils import load_line_label_data, prepare_line_label_data, get_DataLoader, load_model_and_tokenizer, perform_inference, check_gpu_memory

from tqdm import tqdm

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Zero Shot Classification with Llama2-Chat")
    parser.add_argument("--job_id", type=str, default="unknown", help="Job ID")
    parser.add_argument("--model_name", type=str, default="medbert-512", help="Name of model to be used. Defaults to medbert-512. Must be saved in the path: paths.MODEL_PATH/model_name")
    parser.add_argument("--peft_model_name", type=str, help="PEFT model for which to perform inference. Must be saved in the path: paths.MODEL_PATH/model_name. Must be compatible with base model.")
    parser.add_argument("--quantization", type=str, default=None, help="Quantization. Must be one of 4bit, bfloat16 or float16. Defaults to None")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch Size. Defaults to 4")
    parser.add_argument("--split", type=str, default="test", help="Split. Must be one of train, validation or test. Defaults to test")
    parser.add_argument("--task_type", type=str, default="class", help="Task Type. Must be one of class or clm. Defaults to class")

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
    PEFT_MODEL_NAME = args.peft_model_name
    QUANTIZATION = args.quantization
    BATCH_SIZE = args.batch_size
    SPLIT = args.split
    TASK_TYPE = args.task_type
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Check GPU Memory
    check_gpu_memory()

    # Load data
    df = load_line_label_data()
    if TASK_TYPE == "class":
        NUM_LABELS = len(set(df['train']["labels"]))

    # Load Model and Tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name=MODEL_NAME, quantization=QUANTIZATION, num_labels=NUM_LABELS)

    if PEFT_MODEL_NAME:
        model = PeftModel.from_pretrained(model, paths.MODEL_PATH/PEFT_MODEL_NAME)

    print("Loaded Model and Tokenizer")

    # Prepare Data
    truncation_size = tokenizer.model_max_length
    
    encoded_dataset = prepare_line_label_data(df, tokenizer, truncation_size)

    # Get DataLoaders
    dataloader = get_DataLoader(encoded_dataset[SPLIT], tokenizer, batch_size=BATCH_SIZE, shuffle=False)

    # Perform Inference
    inference_results = perform_inference(model, dataloader, device)

    saving_model_name = PEFT_MODEL_NAME if PEFT_MODEL_NAME else MODEL_NAME

    # Save Inference Results
    print("Saving Inference Results at:", paths.RESULTS_PATH/"line-label"/f"{saving_model_name}_{SPLIT}.pt")
    torch.save(inference_results, paths.RESULTS_PATH/"line-label"/f"{saving_model_name}_{SPLIT}.pt")

    return

if __name__ == "__main__":
    main()


