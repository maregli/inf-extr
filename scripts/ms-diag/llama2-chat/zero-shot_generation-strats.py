
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


BASE_PROMPT = "<s>[INST]\n<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt}[/INST]\n\n{answer_init}"
SYSTEM_PROMP = "Is the MS diagnosis in the text of type \"Sekundär progrediente Multiple Sklerose (SPMS)\", \"primäre progrediente Multiple Sklerose (PPMS)\" or \"schubförmig remittierende Multiple Sklerose (RRMS)\"?"
ANSWER_INIT = "Based on the information provided in the text, the most likely diagnosis for the patient is: "

BATCH_SIZE = 4


def parse_args():
    parser = argparse.ArgumentParser(description="Zero Shot Classification with Llama2-Chat")
    parser.add_argument("--job_id", type=str, default="unknown", help="Job ID")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Path to the model. Defaults to llama2-chat")
    parser.add_argument("--quantization", type=str, default=QUANTIZATION, help="Quantization. Must be one of 4bit or bfloat16. Defaults to 4bit")
    parser.add_argument("--base_prompt", type=str, default=BASE_PROMPT, help="Base Prompt, must contain {system_prompt}, {user_prompt} and {answer_init}")
    parser.add_argument("--system_prompt", type=str, default=SYSTEM_PROMP, help="System Prompt")
    parser.add_argument("--answer_init", type=str, default=ANSWER_INIT, help="Answer Initialization for model")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch Size. Defaults to 4")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS, help="Maximum number of new tokens to be generated. Defaults to 20")

    
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

def prepare_data(df:DatasetDict, tokenizer:AutoTokenizer, split:str="all", truncation_size:int = 300)->list[str]:
    """Returns a list of input texts for the classification task
    
    Args:
        df (DatasetDict): Dataset dictionary
        tokenizer (AutoTokenizer): Tokenizer
        split (str, optional): Split. Must be one of train, validation, test or all. Defaults to "all".
        truncation_size (int, optional): Truncation size. Defaults to 300.
        
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
            pass
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

def get_DataLoader(tokens:list[dict], tokenizer:AutoTokenizer, batch_size:int = BATCH_SIZE, padding:bool = True)->DataLoader:
    """Returns a DataLoader for the given dataset dictionary
    
    Args:
        tokens (list[dict]): List of tokenized texts. One dictionary per text with keys input_ids and attention_mask.
        tokenizer (AutoTokenizer): Tokenizer
        batch_size (int, optional): Batch size. Defaults to global BATCH_SIZE.
        padding (bool, optional): Padding. Defaults to True.
        
    Returns:
        DataLoader: Returns a DataLoader for the given dataset dictionary
    """

    # Default collate function 
    collate_fn = DataCollatorWithPadding(tokenizer, padding=padding)

    dataloader = torch.utils.data.DataLoader(dataset=tokens, collate_fn=collate_fn, batch_size=batch_size, shuffle = False) 

    return dataloader

def generate_outputs(model:AutoModelForCausalLM, tokenizer:AutoTokenizer, dataloader:DataLoader, generation_config:dict)->list[str]:
    """Generates outputs for the given model, tokenizer and dataloader
    
    Args:
        model (AutoModelForCausalLM): Model
        tokenizer (AutoTokenizer): Tokenizer
        dataloader (DataLoader): DataLoader
        generation_config (dict): Generation Config
        
    Returns:
        list[str]: Returns a list of outputs
    """
    outputs = []

    for idx, batch in enumerate(tqdm.tqdm(dataloader)):
            
        torch.cuda.empty_cache()
        gc.collect()
        
        input_ids = batch["input_ids"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")
        with torch.inference_mode():
            generated_ids = model.generate(input_ids = input_ids, 
                                            attention_mask = attention_mask,
                                            **generation_config).to("cpu")
    
        outputs.append(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
        check_gpu_memory()

    # Free Memory
    del input_ids
    del attention_mask
    del generated_ids
    torch.cuda.empty_cache()
    gc.collect()

    return outputs

def get_generation_config(strategy:str="greedy")->dict:
    """Returns the generation config for the given strategy

    Args:
        strategy (str, optional): Strategy. Must be one of greedy, contrastive, sampling or beam. Defaults to "greedy".

    Returns:
        dict: Returns the generation config for the given strategy
    """

    # Greedy Search Configuration
    greedy_search_config = {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": 20,
        "temperature": 1, 
        "top_p": 1, # 1 means no top_p sampling filter
        "top_k": 0, # 0 means no top_k sampling filter
        "penalty_alpha": 0.0
    }

    # Contrastive Search Configuration
    contrastive_search_config = {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": 20,
        "temperature": 1, 
        "top_p": 1, # 1 means no top_p sampling filter
        "top_k": 4, # 0 means no top_k sampling filter
        "penalty_alpha": 0.6
    }

    # Sampling Configuration
    sampling_config = {
        "do_sample": True,
        "num_beams": 1,
        "max_new_tokens": 20,
        "temperature": 0.7, 
        "top_p": 0.6, # 1 means no top_p sampling filter
        "top_k": 50, # 0 means no top_k sampling filter
        "penalty_alpha": 0.0
    }

    # Beam Search Configuration
    beam_search_config = {
        "do_sample": False,
        "num_beams": 4,
        "max_new_tokens": 20,
        "temperature": 1, 
        "top_p": 1, # 1 means no top_p sampling filter
        "top_k": 0, # 0 means no top_k sampling filter
        "penalty_alpha": 0.0
    }

    if strategy == "greedy":
        return greedy_search_config
    elif strategy == "contrastive":
        return contrastive_search_config
    elif strategy == "sampling":
        return sampling_config
    elif strategy == "beam":
        return beam_search_config
    else:
        raise ValueError("Strategy must be one of greedy, contrastive, sampling or beam")

def main():

    # Parse Arguments
    args = parse_args()

    # Set Arguments
    JOB_ID = args.job_id
    MODEL_PATH = args.model_path
    QUANTIZATION = args.quantization
    BASE_PROMPT = args.base_prompt
    SYSTEM_PROMP = args.system_prompt
    ANSWER_INIT = args.answer_init
    BATCH_SIZE = args.batch_size
    MAX_NEW_TOKENS = args.max_new_tokens
    
    # Iterate over different truncation sizes and strategies
    strategies = ["greedy", "contrastive", "sampling", "beam"]
    truncation_sizes = [300]

    # Load Data, Model and Tokenizer
    df = load_data()

    print("GPU Memory before Model is loaded:\n")
    check_gpu_memory()
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH, quantization=QUANTIZATION)
    
    print("GPU Memory after Model is loaded:\n")
    check_gpu_memory()

    results = None

    for input_size in truncation_sizes:
        print("Starting with input size: {}".format(input_size))

        for strat in strategies:
            print("Starting with strategy: {}".format(strat))

            # Prepare Data
            tokens = prepare_data(df, tokenizer, split="all", truncation_size=input_size)

            # Get DataLoader
            dataloader = get_DataLoader(tokens, tokenizer, batch_size=BATCH_SIZE, padding=True)

            # Get Generation Config
            generation_config = get_generation_config(strategy=strat)

            # Generate Outputs
            outputs = generate_outputs(model, tokenizer, dataloader, generation_config=generation_config)

            # Extract the generated answers
            outputs = list(chain.from_iterable(outputs))
            outputs = [out.split(ANSWER_INIT)[1] for out in outputs]

            col_name = f"truncate_{input_size}_strategy_{strat}"
            
            if results is None:
                results = pd.DataFrame({col_name: outputs})
            else:
                results[col_name] = outputs
            
            break
        break

    file_name = f"ms_diag-llama2-chat_zero-shot_generation-strats_{JOB_ID}.csv"

    results.to_csv(os.path.join(paths.DATA_PATH, file_name), index=False)

    return

if __name__ == "__main__":
    main()


