
from peft import PeftModel

import torch

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src import paths
from src.utils import (load_ms_data, 
                       prepare_ms_data, 
                       get_DataLoader, 
                       load_model_and_tokenizer, 
                       perform_inference, 
                       check_gpu_memory,
                       information_retrieval,
                       ms_label2id,)

from tqdm import tqdm

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Zero Shot Classification with Llama2-Chat")
    parser.add_argument("--job_id", type=str, default="unknown", help="Job ID")
    parser.add_argument("--model_name", type=str, default="medbert-512", help="Name of model to be used. Defaults to medbert-512. Must be saved in the path: paths.MODEL_PATH/model_name")
    parser.add_argument("--retrieval_model_name", type=str, default="line-label_medbert-512_class_pipeline", help="Name of retrieval model to be used. Defaults to line-label_medbert-512_class_pipeline. Must be saved in the path: paths.MODEL_PATH/retrieval_model_name")
    parser.add_argument("--peft_model_name", type=str, help="PEFT model for which to perform inference. Must be saved in the path: paths.MODEL_PATH/model_name. Must be compatible with base model.")
    parser.add_argument("--quantization", type=str, default=None, help="Quantization. Must be one of 4bit, bfloat16 or float16. Defaults to None")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch Size. Defaults to 4")
    parser.add_argument("--data", type=str, default="line", help="Data. Must be one of line or all. Whether dataset consisting of single lines should be used or all text per rid.")
    parser.add_argument("--data_augmentation", type=str, default=None, help="Must be one of None, zero-shot, augmented or original_approach. Defaults to None.")
    parser.add_argument("--split", type=str, default="test", help="Split. Must be one of train, validation or test. Defaults to test")
    parser.add_argument("--task_type", type=str, default="class", help="Task Type. Must be one of class or clm. Defaults to class")
    parser.add_argument("--output_hidden_states", action="store_true", help="Whether to output hidden states. Defaults to False")
    parser.add_argument("--num_labels", type=int, default=3, help="Number of Labels. Defaults to 3")

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
    RETRIEVAL_MODEL_NAME = args.retrieval_model_name
    PEFT_MODEL_NAME = args.peft_model_name
    QUANTIZATION = args.quantization
    BATCH_SIZE = args.batch_size
    DATA = args.data
    DATA_AUGMENTATION = args.data_augmentation
    SPLIT = args.split
    TASK_TYPE = args.task_type
    OUTPUT_HIDDEN_STATES = args.output_hidden_states
    NUM_LABELS = args.num_labels
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Check GPU Memory
    check_gpu_memory()

    # Load data
    df = load_ms_data(data = DATA)

    # Load Model and Tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name=MODEL_NAME, quantization=QUANTIZATION, num_labels=NUM_LABELS)

    if PEFT_MODEL_NAME:
        model = PeftModel.from_pretrained(model, paths.MODEL_PATH/PEFT_MODEL_NAME)

    print("Loaded Model and Tokenizer")

    # Prepare Data    

    # Information Retrieval
    # Load Retrieval Model
    retrieval_model, retrieval_tokenizer = load_model_and_tokenizer(model_name=RETRIEVAL_MODEL_NAME, 
                                            quantization=None, 
                                            num_labels=8,
                                            task_type="class")
    
    # Retrieve relevant information for training and validation
    original_text = df[SPLIT]["text"]
    text = information_retrieval(retrieval_model=retrieval_model,
                                        retrieval_tokenizer=retrieval_tokenizer,
                                        text=df[SPLIT]["text"],
                                        label = "dm")
    
    df[SPLIT] = df[SPLIT].remove_columns("text").add_column("text", text)
    no_ms_idx = [i for i, t in enumerate(text) if len(t) == 0] # Get indices of empty text

    results = {
        "original_text": original_text,
        "retrieved_text": text,
        "rid": df[SPLIT]["rid"],
        "labels": df[SPLIT]["labels"],
    }

    # Remove empty text
    df[SPLIT] = df[SPLIT].filter(lambda example: len(example["text"]) > 0)
    
    del retrieval_model
    del retrieval_tokenizer

    # Prepare Data
    encoded_dataset = prepare_ms_data(df, tokenizer, data_augmentation=DATA_AUGMENTATION, inference_mode=True)
    
    # Get DataLoaders
    dataloader = get_DataLoader(encoded_dataset[SPLIT], tokenizer, batch_size=BATCH_SIZE, shuffle=False)

    # Perform Inference
    inference_results = perform_inference(model, dataloader, device, output_hidden_states=OUTPUT_HIDDEN_STATES)

    preds = [ms_label2id["no_ms"] if i in no_ms_idx else inference_results["preds"].pop(0) for i in range(len(results["original_text"]))]
    logits = [None if i in no_ms_idx else inference_results["logits"].pop(0) for i in range(len(results["original_text"]))]

    if OUTPUT_HIDDEN_STATES:
        last_hidden_state = [None if i in no_ms_idx else inference_results["last_hidden_state"].pop(0) for i in range(len(results["original_text"]))]
        results["last_hidden_state"] = last_hidden_state
    
    # Add predictions to inference results
    results["preds"] = preds
    results["logits"] = logits

    saving_model_name = PEFT_MODEL_NAME if PEFT_MODEL_NAME else MODEL_NAME

    # Save Inference Results
    print("Saving Inference Results at:", paths.RESULTS_PATH/"ms-diag"/f"{saving_model_name}_{SPLIT}.pt")
    torch.save(results, paths.RESULTS_PATH/"ms-diag"/f"{saving_model_name}_{SPLIT}.pt")

    return

if __name__ == "__main__":
    main()


