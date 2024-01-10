from typing import List, Dict, Tuple, Union, Optional

import torch
from torch.utils.data import DataLoader

from peft import prepare_model_for_kbit_training, PeftConfig, PeftModel

from datasets import DatasetDict, Dataset, load_dataset, concatenate_datasets

import numpy as np

import pandas as pd

from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, BitsAndBytesConfig

from tqdm import tqdm

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src import paths

    
# Embedd the data and predict
def model_output(data: Dataset, model: AutoModelForSequenceClassification, batch_size: int = 32, device: str = 'cuda'):
    """
    Embedd the data and predict

    Args:
        data (datasets.arrow_dataset.Dataset): Dataset to embedd
        model (transformers.models.bert.modeling_bert.BertForSequenceClassification): Model to use.
        batch_size (int, optional): Batch size. Defaults to 32.
    """
    # Create dataloader
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    # Embedd data
    embeddings = []
    logits = []
    labels = []
    for batch in tqdm(dataloader):
        input_ids = torch.stack(batch['input_ids'], dim=1).to(device)
        attention_mask = torch.stack(batch['attention_mask'], dim=1).to(device)
        token_type_ids = torch.stack(batch['token_type_ids'], dim=1).to(device)
        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
            embeddings.append(output.hidden_states[-1].cpu())
            logits.append(output.logits.cpu())
            labels.append(torch.stack(batch['labels'], dim = 1))
    return {"embeddings": torch.cat(embeddings, dim=0), "logits": torch.cat(logits, dim=0), "labels": torch.cat(labels, dim = 0)}

def load_model_and_tokenizer(model_name:str, num_labels:int, quantization:str = "4bit")->Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Loads the model and tokenizer from the given path and returns the compiled model and tokenizer.
    
    Args:
        model (str): Model Name. Assumes that the model is saved in the path: paths.MODEL_PATH/model.
        num_labels (int): Number of labels (classes) to predict.
        quantization (str, optional): Quantization. Must be one of 4bit, bfloat16. Defaults to "4bit".

        Returns:
            tuple(AutoModelForSequenceClassification, AutoTokenizer): Returns the model and tokenizer
            
    """

    ### Model
    if quantization == "bfloat16":
        model = AutoModelForSequenceClassification.from_pretrained(paths.MODEL_PATH/model_name, 
                                                                   torch_dtype=torch.bfloat16,
                                                                   num_labels = num_labels,
                                                                   )
    elif quantization == "float16":
        model = AutoModelForSequenceClassification.from_pretrained(paths.MODEL_PATH/model_name, 
                                                                   torch_dtype=torch.float16,
                                                                   num_labels = num_labels)
    elif quantization == "4bit":
        bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                        bnb_4bit_use_double_quant=True,
                                        bnb_4bit_quant_type="nf4",
                                        bnb_4bit_compute_dtype=torch.bfloat16,
                                        )
        model = AutoModelForSequenceClassification.from_pretrained(paths.MODEL_PATH/model_name, 
                                                                   quantization_config=bnb_config,
                                                                   num_labels = num_labels)
        model = prepare_model_for_kbit_training(model)
    else:
        raise ValueError("Quantization must be one of 4bit, bfloat16 or float16")
        
    ### Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(paths.MODEL_PATH/model_name, padding_side="left")

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

    return model, tokenizer


def load_ms_data(data:str="original")->DatasetDict:
    """Loads the data for MS-Diag task and returns the dataset dictionary

    Args:
        data (str, optional): Data. Must be one of original, zero-shot or augmented. Defaults to "original".
    
    Returns:
        DatasetDict: Returns the dataset dictionary
    """

    data_files = {"train": "ms-diag_clean_train.csv", "validation": "ms-diag_clean_val.csv", "test": "ms-diag_clean_test.csv", "augmented": "ms-diag_augmented.csv"}

    df = load_dataset(os.path.join(paths.DATA_PATH_PREPROCESSED,'ms-diag'), data_files = data_files)

    if data == "zero-shot":
                
        def get_artifical_data_for_label(label:str):
            label_dict = {
                "rrms": "relapsing_remitting_multiple_sclerosis",
                "ppms": "primary_progressive_multiple_sclerosis",
                "spms": "secondary_progressive_multiple_sclerosis"
            }
            generated_data = pd.read_csv(paths.DATA_PATH_PREPROCESSED/f'ms-diag/artificial_{label}.csv')
            generated_data["labels"] = label_dict[label]
            generated_data = generated_data[["0", "labels"]].rename(columns = {"0":"text"})

            return generated_data

        def get_artifical_data_all():
            artifical_data = []
            for label in ["rrms", "ppms", "spms"]:
                try: 
                    artifical_data.append(get_artifical_data_for_label(label))
                except:
                    print(f"Could not find data for {label}")
            artifical_data = pd.concat(artifical_data)
            artifical_data = Dataset.from_pandas(artifical_data).remove_columns('__index_level_0__')
            return artifical_data
        
        df["train"] = concatenate_datasets([get_artifical_data_all(), df["train"]])
    
    elif data == "augmented":
        df["train"] = concatenate_datasets([df["augmented"], df["train"]])

    elif data == "original":
        pass
    else:
        raise ValueError("Data must be one of original, zero-shot or augmented")
            
    return df

# MS Label to id
ms_label2id = {'primary_progressive_multiple_sclerosis': 0,
                'relapsing_remitting_multiple_sclerosis': 1,
                'secondary_progressive_multiple_sclerosis': 2}
ms_id2label = {v:k for k,v in ms_label2id.items()}


def prepare_ms_data(df:DatasetDict, tokenizer:AutoTokenizer, is_prompt_tuning:bool = False, num_virtual_tokens:int=20, truncation_size:int = 512)->DatasetDict:
    """Prepares the data for MS-Diag task and returns the dataset

    Args:
        df (DatasetDict): Dataset dictionary
        tokenizer (AutoTokenizer): Tokenizer
        is_prompt_tuning (bool, optional): Whether to use prompt tuning. Defaults to False.
        num_virtual_tokens (int, optional): Number of virtual tokens. Defaults to 20.
        truncation_size (int, optional): Truncation size. Defaults to 512.

    Returns:
        DatasetDict: Returns the dataset
    """
    
    # For Prompt Tuning, we need to add the prefix to the input text
    if is_prompt_tuning:
        assert num_virtual_tokens > 0, "Number of virtual tokens must be greater than 0 for prompt tuning"
        max_length = tokenizer.model_max_length - num_virtual_tokens
    else:
        max_length = tokenizer.model_max_length
        
    truncation_size = min(truncation_size, max_length)

    def tokenize_function(examples):
        outputs = tokenizer(examples["text"], truncation=True, max_length=truncation_size)
        outputs["labels"] = [ms_label2id[label] for label in examples["labels"]]
        return outputs

    encoded_dataset = df.map(tokenize_function, batched=True, remove_columns=["text", "rid", "date"])

    return encoded_dataset


def get_DataLoader(df:Dataset, tokenizer:AutoTokenizer, batch_size:int = 4, shuffle:bool = True)->DataLoader:
    """Returns a DataLoader for the given dataset dictionary
    
    Args:
        df (Dataset): HF Dataset
        tokenizer (AutoTokenizer): Tokenizer
        batch_size (int, optional): Batch size. Defaults to 4.
        shuffle (bool, optional): Shuffle. Defaults to False.
        
    Returns:
        DataLoader: Returns a DataLoader for the given dataset dictionary, with padded batches by DataCollatorWithPadding.
    """

    # Default collate function 
    collate_fn = DataCollatorWithPadding(tokenizer, padding="longest", pad_to_multiple_of=8)

    dataloader = torch.utils.data.DataLoader(dataset=df, collate_fn=collate_fn, batch_size=batch_size, shuffle = shuffle) 

    return dataloader


def perform_inference(model:Union[PeftModel, AutoModelForSequenceClassification], dataloader:DataLoader, device:torch.device)->dict[str, Union[List[torch.Tensor], List[int]]]:
    """Performs inference on the given dataloader using the given model and returns the last hidden states, labels and predictions.

    Args:
        model (Union[PeftModel, AutoModelForSequenceClassification]): Model to use for inference.
        dataloader (DataLoader): DataLoader to use for inference.
        device (torch.device): Device to use for inference.

    Returns:
        dict[str, Union[List[torch.Tensor], List[int]]]: Returns a dictionary containing the last hidden states List[torch.Tensor], labels List[int]  and predictions List[int].
    """
    model.eval()

    with torch.no_grad():
        test_preds = []
        label_list = []
        last_hidden_states = []

        for batch in tqdm(dataloader):
            batch.to(device)
            
            outputs = model(**batch, output_hidden_states=True)
            logits = outputs.logits.to("cpu")
            labels = batch['labels'].to("cpu")
            
            predictions = logits.argmax(dim=-1)
            test_preds.extend(predictions.tolist())
            
            label_list.extend(labels.tolist())
            last_hidden_states.extend(outputs.hidden_states[-1].to("cpu"))

    return {"last_hidden_state": last_hidden_states, 
            "labels": label_list,
            "test_preds": test_preds}   


def check_gpu_memory()->None:
    """Checks the GPU memory and prints the results.
    """
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