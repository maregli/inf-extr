
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorWithPadding

from datasets import DatasetDict, load_dataset

import torch
from torch.utils.data import DataLoader
import gc

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src import paths

from itertools import chain

import pandas as pd

import tqdm

import argparse
from typing import Tuple

MODEL_PATH = paths.MODEL_PATH/'llama2-chat'
QUANTIZATION = "4bit"

SPLIT = "train"

BASE_PROMPT = "<s>[INST]\n<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt}[/INST]\n\n{answer_init}"
SYSTEM_PROMP = "Is the MS diagnosis in the text of type \"Sekundär progrediente Multiple Sklerose (SPMS)\", \"primäre progrediente Multiple Sklerose (PPMS)\" or \"schubförmig remittierende Multiple Sklerose (RRMS)\"?"
ANSWER_INIT = "Based on the information provided in the text, the most likely diagnosis for the patient is: "
TRUNCATION_SIZE = 300

BATCH_SIZE = 4
DO_SAMPLE = False
NUM_BEAMS = 1
MAX_NEW_TOKENS = 20
TEMPERATURE = 1
TOP_P = 1
TOP_K = 4
PENALTY_ALPHA = 0.0

def parse_args():
    parser = argparse.ArgumentParser(description="Zero Shot Classification with Llama2-Chat")
    parser.add_argument("--job_id", type=str, default="unknown", help="Job ID")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Path to the model. Defaults to llama2-chat")
    parser.add_argument("--quantization", type=str, default=QUANTIZATION, help="Quantization. Must be one of 4bit or bfloat16. Defaults to 4bit")
    parser.add_argument("--split", type=str, default=SPLIT, help="Data Split. Must be one of train, validation, test or all. Defaults to train")
    parser.add_argument("--base_prompt", type=str, default=BASE_PROMPT, help="Base Prompt, must contain {system_prompt}, {user_prompt} and {answer_init}")
    parser.add_argument("--system_prompt", type=str, default=SYSTEM_PROMP, help="System Prompt")
    parser.add_argument("--answer_init", type=str, default=ANSWER_INIT, help="Answer Initialization for model")
    parser.add_argument("--truncation_size", type=int, default=TRUNCATION_SIZE, help="Truncation Size of the input text. Defaults to 300")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch Size. Defaults to 4")
    parser.add_argument("--do_sample", type=str, default=DO_SAMPLE, help="Do Sampling. Defaults to False")
    parser.add_argument("--num_beams", type=int, default=NUM_BEAMS, help="Number of Beams for Beam Search. Defaults to 1")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS, help="Maximum number of new tokens to be generated. Defaults to 20")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="Temperature for sampling. Defaults to 1")
    parser.add_argument("--top_p", type=float, default=TOP_P, help="Top p for sampling. Defaults to 1")
    parser.add_argument("--top_k", type=int, default=0, help="Top k for sampling. Defaults to 4")
    parser.add_argument("--penalty_alpha", type=float, default=0.0, help="Penalty Alpha for Beam Search. Defaults to 0.0")
    
    args = parser.parse_args()

    # Print or log the parsed arguments
    print("Parsed Arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    return args


def check_gpu_memory():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for gpu_id in range(num_gpus):
            free_mem, total_mem = torch.cuda.mem_get_info(gpu_id)
            gpu_properties = torch.cuda.get_device_properties(gpu_id)
            print(f"GPU {gpu_id}: {gpu_properties.name}")
            print(f"   Total Memory: {total_mem / (1024 ** 3):.2f} GB")
            print(f"   Free Memory: {free_mem / (1024 ** 3):.2f} GB")
            print(f"   Allocated Memory : {torch.cuda.memory_allocated(gpu_id) / (1024 ** 3):.2f} GB")
            print(f"   Reserved Memory : {torch.cuda.memory_reserved(gpu_id) / (1024 ** 3):.2f} GB")
    else:
        print("No GPU available.")


# Load Model and tokenizer

def load_model_and_tokenizer(model_path:os.PathLike, quantization:str = QUANTIZATION)->Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Loads the model and tokenizer from the given path and returns the compiled model and tokenizer.
    
    Args:
        model_path (os.PathLike): Path to the model
        quantization (str, optional): Quantization. Must be one of 4bit or bfloat16. Defaults to QUANTIZATION.

        Returns:
            tuple(AutoModelForCausalLM, AutoTokenizer): Returns the compiled model and tokenizer
            
    """
    # ### Model
    if quantization == "bfloat16":
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
    elif quantization == "4bit":
        bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                        bnb_4bit_use_double_quant=True,
                                        bnb_4bit_quant_type="nf4",
                                        bnb_4bit_compute_dtype=torch.bfloat16,
                                        )
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", quantization_config=bnb_config)
    else:
        raise ValueError("Quantization must be one of 4bit or bfloat16")
    
    ### Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")

    # Check if the pad token is already in the tokenizer vocabulary
    if '<pad>' not in tokenizer.get_vocab():
        # Add the pad token
        tokenizer.add_special_tokens({"pad_token":"<pad>"})
    

    #Resize the embeddings
    model.resize_token_embeddings(len(tokenizer))

    #Configure the pad token in the model
    model.config.pad_token_id = tokenizer.pad_token_id

    # Check if they are equal
    assert model.config.pad_token_id == tokenizer.pad_token_id, "The model's pad token ID does not match the tokenizer's pad token ID!"

    # Print the pad token ids
    print('Tokenizer pad token ID:', tokenizer.pad_token_id)
    print('Model pad token ID:', model.config.pad_token_id)
    print('Model config pad token ID:', model.config.pad_token_id)
    print("Vocabulary Size with Pad Token: ", len(tokenizer))

    return torch.compile(model), tokenizer # Compile Model for faster inference. # To-Do https://pytorch.org/blog/pytorch-compile-to-speed-up-inference/


def load_data()->DatasetDict:
    """Loads the data for MS-Diag task and returns the dataset dictionary
    
    Returns:
        DatasetDict: Returns the dataset dictionary
    """

    data_files = {"train": "ms-diag_clean_train.csv", "validation": "ms-diag_clean_val.csv", "test": "ms-diag_clean_test.csv"}

    df = load_dataset(os.path.join(paths.DATA_PATH_PREPROCESSED,'ms-diag'), data_files = data_files)
    
    return df

def prepare_data(df:DatasetDict, tokenizer:AutoTokenizer, split:str=SPLIT, truncation_size:int = TRUNCATION_SIZE)->list[str]:
    """Returns a list of input texts for the classification task
    
    Args:
        df (DatasetDict): Dataset dictionary
        tokenizer (AutoTokenizer): Tokenizer
        split (str, optional): Split. Defaults to SPLIT.
        truncation_size (int, optional): Truncation size. Defaults to TRUNCATION_SIZE.
        
    Returns:
        list(str): Returns a list of input texts for the classification task
    """

    def format_prompt(text:str)->str:
        """Truncates the text to the given truncation size and formats the prompt.
        
        Args:
            text (str): Text
        
        Returns:
            str: Returns the formatted prompt
        """
        if len(text) > truncation_size:
            text = text[:truncation_size]
        else:
            text = text
        input = BASE_PROMPT.format(system_prompt = SYSTEM_PROMP,
                                user_prompt = text,
                                answer_init = ANSWER_INIT)

        return input

    
    # Tokenize the text
    if split == "all":
        text = df["train"]["text"] + df["validation"]["text"] + df["test"]["text"]
    else:
        text = df[split]["text"]

    tokens = [tokenizer(format_prompt(t)) for t in text]

    return tokens

def get_DataLoader(tokens:list[str], tokenizer:AutoTokenizer, batch_size:int = BATCH_SIZE, padding:bool = True)->DataLoader:
    """Returns a DataLoader for the given dataset dictionary
    
    Args:
        tokens (List(str)): List of input texts
        tokenizer (AutoTokenizer): Tokenizer
        batch_size (int, optional): Batch size. Defaults to BATCH_SIZE.
        padding (bool, optional): Padding. Defaults to True.
        
    Returns:
        DataLoader: Returns a DataLoader for the given dataset dictionary
    """

    # Default collate function 
    collate_fn = DataCollatorWithPadding(tokenizer, padding=padding)

    dataloader = torch.utils.data.DataLoader(dataset=tokens, collate_fn=collate_fn, batch_size=batch_size, shuffle = False) 

    return dataloader

def main():

    args = parse_args()
    JOB_ID = args.job_id
    SPLIT = args.split
    MODEL_PATH = args.model_path
    QUANTIZATION = args.quantization
    BASE_PROMPT = args.base_prompt
    SYSTEM_PROMP = args.system_prompt
    ANSWER_INIT = args.answer_init
    BATCH_SIZE = args.batch_size
    DO_SAMPLE = True if args.do_sample == "True" else False
    TRUNCATION_SIZE = args.truncation_size
    NUM_BEAMS = args.num_beams
    MAX_NEW_TOKENS = args.max_new_tokens
    TEMPERATURE = args.temperature
    TOP_P = args.top_p
    TOP_K = args.top_k

    # Load Data, Model and Tokenizer
    df = load_data()

    print("GPU Memory before Model is loaded:\n")
    check_gpu_memory()
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH, quantization=QUANTIZATION)
    print("GPU Memory after Model is loaded:\n")
    check_gpu_memory()

    # Prepare Data
    tokens = prepare_data(df, tokenizer, split=SPLIT, truncation_size=TRUNCATION_SIZE)

    # Get DataLoader
    dataloader = get_DataLoader(tokens, tokenizer, batch_size=BATCH_SIZE, padding=True)

    # Inference
    outputs = []

    for idx, batch in enumerate(tqdm.tqdm(dataloader)):
            
        torch.cuda.empty_cache()
        gc.collect()
        
        input_ids = batch["input_ids"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")
        with torch.inference_mode():
            generated_ids = model.generate(input_ids = input_ids, 
                                           attention_mask = attention_mask,
                                            max_new_tokens=MAX_NEW_TOKENS, 
                                            num_beams=NUM_BEAMS, 
                                            do_sample=DO_SAMPLE, 
                                            temperature = TEMPERATURE, 
                                            num_return_sequences = 1, 
                                            top_p = TOP_P,
                                            top_k = TOP_K,
                                            penalty_alpha = PENALTY_ALPHA).to("cpu")
    
        outputs.append(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
        print("Memory after batch {}:\n".format(idx))
        check_gpu_memory()

    # Save results
    outputs = list(chain.from_iterable(outputs))
    results = [out.split(ANSWER_INIT)[1] for out in outputs]
    
    # Add Arguments as last row to the results
    results.append(str(args))

    file_name = f"ms_diag-llama2-chat_zero-shot_{JOB_ID}.csv"
    pd.Series(results).to_csv(paths.RESULTS_PATH/file_name)

    return

if __name__ == "__main__":
    main()


