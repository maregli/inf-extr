
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
QUANTIZATION = True

SPLIT = "train"

BASE_PROMPT = "<s>[INST]\n<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt}[/INST]\n\n{answer_init}"
SYSTEM_PROMP = "Is the MS diagnosis in the text of type \"Sekundär progrediente Multiple Sklerose (SPMS)\", \"primäre progrediente Multiple Sklerose (PPMS)\" or \"schubförmig remittierende Multiple Sklerose (RRMS)\"?"
ANSWER_INIT = "Based on the information provided in the text, the most likely diagnosis for the patient is: "
TRUNCATION_SIZE = 300

BATCH_SIZE = 1
NUM_BEAMS = 1
MAX_NEW_TOKENS = 1
TEMPERATURE = 0.9
TOP_P = 0.6

def parse_args():
    parser = argparse.ArgumentParser(description="Zero Shot Classification with Llama2-Chat")
    parser.add_argument("--job_id", type=str, default="unknown", help="Job ID")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Path to the model")
    parser.add_argument("--quantization", type=bool, default=QUANTIZATION, help="Quantization")
    parser.add_argument("--split", type=str, default=SPLIT, help="Data Split. Must be one of train, validation or test. Defaults to train")
    parser.add_argument("--base_prompt", type=str, default=BASE_PROMPT, help="Base Prompt, must contain {system_prompt}, {user_prompt} and {answer_init}")
    parser.add_argument("--system_prompt", type=str, default=SYSTEM_PROMP, help="System Prompt")
    parser.add_argument("--answer_init", type=str, default=ANSWER_INIT, help="Answer Initialization for model")
    parser.add_argument("--truncation_size", type=int, default=TRUNCATION_SIZE, help="Truncation Size of the input text")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch Size")
    parser.add_argument("--num_beams", type=int, default=NUM_BEAMS, help="Number of Beams for Beam Search")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS, help="Maximum number of new tokens to be generated")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=TOP_P, help="Top p for sampling")
    
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

def load_model_and_tokenizer(model_path:os.PathLike, quantization_config:BitsAndBytesConfig = None)->Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Loads the model and tokenizer from the given path and returns the compiled model and tokenizer.
    
    Args:
        model_path (os.PathLike): Path to the model
        quantization_config (BitsAndBytesConfig, optional): Quantization Config. Defaults to None, in which case model is loaded in bfloat16.
        
        Returns:
            tuple(AutoModelForCausalLM, AutoTokenizer): Returns the compiled model and tokenizer
            
    """
    ### Model
    if QUANTIZATION == False:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
    else:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                        bnb_4bit_use_double_quant=True,
                                        bnb_4bit_quant_type="nf4",
                                        bnb_4bit_compute_dtype=torch.bfloat16,
                                        )
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", quantization_config=bnb_config)
    
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
        text = text[:truncation_size]
        input = BASE_PROMPT.format(system_prompt = SYSTEM_PROMP,
                                user_prompt = text,
                                answer_init = ANSWER_INIT)

        return input

    
    # Tokenize the text
    tokens = [tokenizer(format_prompt(t)) for t in df[split]["text"]]

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
    TRUNCATION_SIZE = args.truncation_size
    NUM_BEAMS = args.num_beams
    MAX_NEW_TOKENS = args.max_new_tokens
    TEMPERATURE = args.temperature
    TOP_P = args.top_p

    # Load Data, Model and Tokenizer
    df = load_data()

    print("GPU Memory before Model is loaded:\n")
    check_gpu_memory()
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH, quantization_config=QUANTIZATION)
    print("GPU Memory after Model is loaded:\n")
    check_gpu_memory()

    # Prepare Data
    tokens = prepare_data(df, tokenizer, split=SPLIT, truncation_size=TRUNCATION_SIZE)

    # Get DataLoader
    dataloader = get_DataLoader(tokens, tokenizer, batch_size=BATCH_SIZE, padding=True)

    # Inference
    outputs = []

    for batch in tqdm.tqdm(dataloader):
            
        torch.cuda.empty_cache()
        gc.collect()
        
        input_ids = batch["input_ids"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")
        with torch.inference_mode():
            generated_ids = model.generate(input_ids = input_ids, 
                                           attention_mask = attention_mask,
                                            max_new_tokens=MAX_NEW_TOKENS, 
                                            num_beams=NUM_BEAMS, 
                                            do_sample=True, 
                                            temperature = TEMPERATURE, 
                                            num_return_sequences = 1, 
                                            top_p = TOP_P).to("cpu")
    
        outputs.append(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
        break

    # Save results
    outputs = list(chain.from_iterable(outputs))
    results = [out.split(ANSWER_INIT)[1] for out in outputs]

    file_name = f"ms_diag-llama2-chat_zero-shot_{JOB_ID}.csv"
    pd.Series(results).to_csv(paths.RESULTS_PATH/file_name)

    return "Done"

if __name__ == "__main__":
    main()


