
import torch
from torch.utils.data import DataLoader

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src import paths
from src.utils import (load_model_and_tokenizer,  
                       check_gpu_memory, 
)

import argparse

from transformers import  AutoTokenizer, AutoModel, DataCollatorWithPadding

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Zero Shot Classification with Llama2-Chat")
    parser.add_argument("--job_id", type=str, default="unknown", help="Job ID")
    parser.add_argument("--model_name", type=str, default="medbert-512", help="Name of base model to be used. Defaults to medbert. Must be saved in the path: paths.MODEL_PATH/model_name")
    parser.add_argument("--quantization", type=str, default=None, help="Quantization. Must be one of 4bit, bfloat16, float16 or None. Defaults to None")
    parser.add_argument("--results_files", nargs="+", type=str, default=None, help="List of results file. Defaults to None. Files be saved under paths.RESULTS_PATH/ms-diag/results_file and contain a dict with key:  prediction")

    args = parser.parse_args()

    # Print or log the parsed arguments
    print("Parsed Arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    return args

def get_hidden_state(results:dict, model:AutoModel, tokenizer:AutoTokenizer, device:torch.device=torch.device("cpu"))->list[torch.Tensor]:
    """Get hidden state of last layer of model for each prediction in results
    
    Args:
        results (dict): results of prompting return with keys report, prediction, last_hidden_states, input_lengths, whole_prompt
        model (AutoModel): model
        tokenizer (AutoTokenizer): tokenizer
        device (torch.device): device. Defaults to torch.device("cpu").
        
    Returns:
        results (dict): results of prompting return with keys report, prediction, last_hidden_states, input_lengths, whole_prompt, encodings
            
            
    """

    inputs = tokenizer(results["prediction"], add_special_tokens = False)
    collate_fn = DataCollatorWithPadding(tokenizer = tokenizer, padding = "longest")

    dataset = torch.utils.data.TensorDataset(inputs["input_ids"], inputs["attention_mask"])
    dataloader = DataLoader(dataset = dataset, batch_size = 16, collate_fn = collate_fn)
    
    encodings = []

    for batch in tqdm(dataloader):
        batch.to(device)

        with torch.no_grad():
            outputs = model(**input, output_hidden_states = True)

        last_hidden_state = outputs["hidden_states"][-1]
        # Take last token of last hidden state
        last_hidden_state = last_hidden_state[:, -1, :]
        encodings.append(last_hidden_state)
        del last_hidden_state
        del outputs
        del input
        torch.cuda.empty_cache()
    
    results["last_hidden_states"] = torch.cat(encodings, dim=0)

    return results

def main()->None:

    args = parse_args()
    JOB_ID = args.job_id
    MODEL_NAME = args.model_name
    QUANTIZATION = args.quantization
    RESULTS_FILES = args.results_files

    assert RESULTS_FILES is not None, "Results file must be specified"

    # Check GPU Memory
    check_gpu_memory()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model and Tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name = MODEL_NAME,
                                                task_type = "clm",
                                                quantization = QUANTIZATION,
                                                )
    model.config.use_cache = False
    
    check_gpu_memory()

    print("Loaded Model and Tokenizer")

    for RESULTS_FILE in RESULTS_FILES:
        print(f"Processing {RESULTS_FILE}")

        # Load Results
        results = torch.load(paths.RESULTS_PATH/"ms-diag"/RESULTS_FILE)

        results = get_hidden_state(results = results,
                                model = model,
                                tokenizer = tokenizer,
                                device = device,
                                )
        file_name = RESULTS_FILE.split(".")[0] + "_hidden_state.pt"
        torch.save(results, paths.RESULTS_PATH/"ms-diag"/file_name)

    return

if __name__ == "__main__":
    main()


