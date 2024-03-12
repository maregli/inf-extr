import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src import paths

from typing import List, Dict, Tuple, Union, Optional, Callable

import torch
from torch.utils.data import DataLoader

from peft import PeftModel, LoraConfig, PromptEncoderConfig, PromptTuningConfig, PrefixTuningConfig, PromptTuningInit, PeftConfig, prepare_model_for_kbit_training

from datasets import DatasetDict, Dataset, load_dataset, concatenate_datasets, interleave_datasets

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import seaborn as sns

import outlines
from outlines import samplers, models
from outlines.generate import SequenceGenerator

from transformers import (AutoModelForSequenceClassification,
                          AutoModelForCausalLM,
                          AutoModelForTokenClassification, 
                          AutoTokenizer, 
                          DataCollatorWithPadding, 
                          BitsAndBytesConfig, 
                          get_linear_schedule_with_warmup,
                          )

from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, 
                             f1_score, 
                             precision_score, 
                             recall_score, 
                             ConfusionMatrixDisplay, 
                             precision_recall_fscore_support,
                             classification_report,
                             confusion_matrix,)


from umap import UMAP

from collections import Counter

from pydantic import BaseModel

from enum import Enum

###########################################################################################################
# Model and Tokenizer
###########################################################################################################

def load_model_and_tokenizer(model_name:str, 
                             num_labels:int = 0, 
                             task_type:str="class", 
                             quantization: Optional[str] = None, 
                             attn_implementation: Optional[str] = None,
                             truncation_side:str = "right",
                             )->Tuple[Union[AutoModelForSequenceClassification, AutoModelForCausalLM, AutoModelForTokenClassification], AutoTokenizer]:
    """Loads the model and tokenizer from the given path and returns the compiled model and tokenizer.
    
    Args:
        model (str): Model Name. Assumes that the model is saved in the path: paths.MODEL_PATH/model.
        num_labels (int): Number of labels (classes) to predict. Defaults to 0.
        task_type (str): Task Type. Must be one of class, token, clm or outlines. Defaults to "class".
        quantization (str, optional): Quantization. Can be one of 4bit, bfloat16 or float16. Defaults to None.
        attn_implementation (str, optional): To implement Flash Attention 2 provide "flash_attention_2". Defaults to None.
        truncation_side (str, optional): Truncation Side. Defaults to "right".

        Returns:
            Tuple[Union[AutoModelForSequenceClassification, AutoModelForCausalLM, AutoModelForTokenClassification], AutoTokenizer]:
              Returns the model and tokenizer.
            
    """

    if quantization == "4bit":
        quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                            bnb_4bit_use_double_quant=True,
                                            bnb_4bit_quant_type="nf4",
                                            bnb_4bit_compute_dtype=torch.bfloat16,
                                            )
        torch_dtype = None

    elif quantization == "bfloat16":
        quantization_config = None
        torch_dtype = torch.bfloat16

    elif quantization == "float16":
        quantization_config = None
        torch_dtype = torch.float16

    else:
        quantization_config = None
        torch_dtype = None

    config_kwargs = {
        "pretrained_model_name_or_path": paths.MODEL_PATH/model_name,
        "quantization_config":quantization_config,
        "attn_implementation": attn_implementation,
        "torch_dtype": torch_dtype,
    }

    if task_type == "class":
        assert num_labels > 0, "Number of labels must be greater than 0 for classification task"
        model = AutoModelForSequenceClassification.from_pretrained(**config_kwargs, num_labels = num_labels)
        
    elif task_type == "clm":
        model = AutoModelForCausalLM.from_pretrained(**config_kwargs,
                                                    device_map = "auto")
    
    elif task_type == "token":
        # As the only task that needs this is line labelling, we can hardcode the number of labels and id2label mappings
        num_labels = len(line_label_token_label2id)
        model = AutoModelForTokenClassification.from_pretrained(**config_kwargs, 
                                                                num_labels=num_labels, 
                                                                id2label=line_label_token_id2label, 
                                                                label2id=line_label_token_label2id)
    elif task_type == "outlines":
        # Outlines will always work with clm
        model = models.transformers(config_kwargs.pop("pretrained_model_name_or_path"),
                                    device="cuda", 
                                   model_kwargs = config_kwargs,)
        

    ### Tokenizer
    # If model is bert model, use bert tokenizer
    if "bert" in model_name:
        padding_side = "right"
    else:
        padding_side = "left"

    tokenizer = AutoTokenizer.from_pretrained(paths.MODEL_PATH/model_name, padding_side=padding_side, truncation_side=truncation_side)

    if not task_type == "outlines":
        # Set max length to max of 2048 and model max length, can cause issues with int too large
        tokenizer.model_max_length = min(tokenizer.model_max_length, 4096)
    
        # Check if the pad token is already in the tokenizer vocabulary
        if tokenizer.pad_token_id is None:
            # Add the pad token
            tokenizer.add_special_tokens({"pad_token":"<pad>"})
        
        # If task is token classification add special line break token
        if task_type == "token" and "[BRK]" not in tokenizer.special_tokens_map.get("additional_special_tokens", []):
            tokenizer.add_special_tokens({"additional_special_tokens":["[BRK]"]})
            print("Added special token [BRK] to tokenizer")
        
    
        #Resize the embeddings
        model.resize_token_embeddings(len(tokenizer))
    
        #Configure the pad token in the model
        model.config.pad_token_id = tokenizer.pad_token_id
    
        # Check if they are equal
        assert model.config.pad_token_id == tokenizer.pad_token_id, "The model's pad token ID does not match the tokenizer's pad token ID!"
    
        # Print the pad token ids
        print('Tokenizer pad token ID:', tokenizer.pad_token_id)
        print("Tokenizer special tokens:", tokenizer.special_tokens_map)
        print('Model pad token ID:', model.config.pad_token_id)

    return model, tokenizer


def select_peft_config(model:AutoModelForSequenceClassification, peft_type:str)->PeftConfig:
    """Selects the peft config for the given model and peft type

    Args:
        model (AutoModelForSequenceClassification): Model
        peft_type (str): Peft Type. Must be one of lora, prefix, ptune or prompt

    Returns:
        PeftConfig: Returns the peft config
    """

    if peft_type == "lora":
        peft_config = LoraConfig(lora_alpha=16,
                                 lora_dropout=0.1,
                                 r=8,
                                 bias="none",
                                 task_type="SEQ_CLS"
                                 )
    elif peft_type == "prefix":
        config = {
            "peft_type": "PREFIX_TUNING",
            "task_type": "SEQ_CLS",
            "inference_mode": False,
            "num_virtual_tokens": 0,
            "token_dim": model.config.hidden_size,
            "num_transformer_submodules": 1,
            "num_attention_heads": model.config.num_attention_heads,
            "num_layers": model.config.num_hidden_layers,
            "encoder_hidden_size": 128,
            "prefix_projection": True,
            }
        peft_config = PrefixTuningConfig(**config)
    
    elif peft_type == "ptune":
        peft_config = PromptEncoderConfig(task_type="SEQ_CLS", 
                                          num_virtual_tokens=20, 
                                          encoder_hidden_size=128, 
                                          encoder_dropout=0.1)
        
    elif peft_type == "prompt":
        peft_config = PromptTuningConfig(task_type="SEQ_CLS",
                                         prompt_tuning_init=PromptTuningInit.TEXT,
                                         num_virtual_tokens=20,
                                         prompt_tuning_init_text="Klassifiziere als primär, sekundär oder schubförmige MS",
                                         tokenizer_name_or_path=os.path.join(paths.MODEL_PATH/'llama2-chat'),
                                         )
    else:
        raise ValueError("PEFT Type must be one of lora, prefix, ptune or prompt")
    
    return peft_config

########################################################################################################### 
# Data
###########################################################################################################

def load_line_label_data(version:str="base")->DatasetDict:
    """Loads the data for Line Label task and returns the dataset dictionary

    Args:
        version (str, optional): Version. Must be one of base or pipeline. Defaults to "base".

    Returns:
        DatasetDict: Returns the dataset dictionary
    """

    if version == "base":
        try:
            dataset = DatasetDict.load_from_disk(paths.DATA_PATH_PREPROCESSED/'line-label/line-label_clean_dataset')
        except:
            print("Could not find the dataset. Try running the preprocessing notebook: notebooks/01_classifying_text_lines.ipynb")
    elif version == "pipeline":
        try:
            dataset = DatasetDict.load_from_disk(paths.DATA_PATH_PREPROCESSED/'line-label/line-label_clean_dataset_pipeline')
        except:
            print("Could not find the dataset. Try running the preprocessing notebook: notebooks/01_classifying_text_lines.ipynb")
    return dataset



def prepare_line_label_data(dataset:DatasetDict, tokenizer:AutoTokenizer, truncation_size:int = 512, inference_mode = False)->DatasetDict:
    """Prepares the data for Line Label task and returns the dataset

    Args:
        dataset (DatasetDict): Dataset dictionary for Line Label task.
        tokenizer (AutoTokenizer): Tokenizer.
        truncation_size (int, optional): Truncation size. Defaults to 512.

    Returns:
        DatasetDict: Returns the dataset
    """

    max_length = tokenizer.model_max_length
    truncation_size = min(truncation_size, max_length)

    def tokenize_function(examples):
        outputs = tokenizer(examples["text"], truncation=True, max_length=truncation_size)

        if not inference_mode:
            outputs["labels"] = examples["labels"] # During inference labels are removed because None labels cause issues

        return outputs

    encoded_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

    return encoded_dataset


def load_line_label_token_data():
    """Loads the data for Line Label token class task and returns the dataset dictionary
    
    Returns:
        DatasetDict: Returns the dataset dictionary
    """
    dataset_token = DatasetDict.load_from_disk(paths.DATA_PATH_PREPROCESSED/"line-label/line-label_for_token_classification")
    return dataset_token


def tokenize_and_align_labels(examples:dict, tokenizer:AutoTokenizer):
    """Tokenizes the given examples and aligns the labels for the line classification task. Expects the examples to have the columns: text, ner_tags.
    The ner_tags must be in int format. This function is used by the HuggingFace Dataset.map() function.

    Args:
        examples (dict): Dictionary containing the examples
        tokenizer (AutoTokenizer): Tokenizer

    Returns:
        dict: Returns the tokenized examples
    """

    tokenized_inputs = tokenizer(examples["text"], truncation=True, is_split_into_words=True, max_length=tokenizer.model_max_length)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        tokenized_words = tokenized_inputs.tokens(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for tokenized_word, word_idx in zip(tokenized_words, word_ids):  # Set the special tokens to -100.
            if word_idx is None or tokenized_word == "[BRK]":
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs



def load_ms_data(data:str="line")->DatasetDict:
    """Loads the data for MS-Diag task and returns the dataset dictionary

    Args:
        data (str, optional): Data. Must be one of "line", "all" or "all_first_line_last". Defaults to "line".
    
    Returns:
        DatasetDict: Returns the dataset dictionary
    """

    df = DatasetDict.load_from_disk(os.path.join(paths.DATA_PATH_PREPROCESSED, f"ms-diag/ms_diag_{data}"))
            
    return df
    

def prepare_ms_data(df:DatasetDict, tokenizer: AutoTokenizer, data_augmentation:str=None, inference_mode:bool = False)->DatasetDict:
    """Prepares the data for MS-Diag task and returns the dataset

    Args:
        df (DatasetDict): Dataset dictionary. Must have keys "train", "val" and "test". Each subdataset must have columns "text" (and "labels" if training).
        tokenizer (AutoTokenizer): Tokenizer
        data_augmentation (str, optional): Data Augmentation. Must be one of None, "zero-shot" or "augmented". Defaults to None.
        inference_mode (bool, optional): Inference Mode, will . Defaults to False.

    Returns:
        DatasetDict: Returns the dataset
    """

    
    if data_augmentation == "zero-shot":
                
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
    
    elif data_augmentation == "augmented":
        df["train"] = concatenate_datasets([df["augmented"], df["train"]])

    elif data_augmentation == "undersample":
        df["train"] = undersample_dataset(df["train"])

    elif data_augmentation == "oversample":
        df["train"] = oversample_dataset(df["train"])

    elif data_augmentation == "original_approach":
        df = df.filter(lambda e: e["labels"] != ms_label2id["no_ms"])
        df["train"] = oversample_dataset(df["train"])
    
    elif not data_augmentation:
        pass
    
    else:
        raise ValueError("Data Augmentation must be one of None, zero-shot or augmented")
    
    # For evaluation during training will use oversampling
    if not inference_mode:
        df["val"] = oversample_dataset(df["val"])
    
    def tokenize_function(examples):
        batch = tokenizer(examples["text"], truncation=True, max_length=tokenizer.model_max_length)
        
        if not inference_mode:
            batch["labels"] = examples["labels"]  # During inference labels are removed because None labels cause issues

        return batch
    
    encoded_dataset = df.map(tokenize_function, batched=True, remove_columns=df["train"].column_names)

    return encoded_dataset

def undersample_dataset(df:Dataset)->Dataset:
    """Undersamples the given dataset for the given label to the desired length. Assumes that the dataset has a column "labels".

    Args:
        dataset (Dataset): Dataset
        label (str): Label
        desired_len (int, optional): Desired Length. Defaults to 100.

    Returns:
        Dataset: Returns the undersampled dataset
    """
    dfs = [df.filter(lambda e: e["labels"] == label) for label in set(df["labels"])]
    df = interleave_datasets(dfs, seed=42)

    return df

def oversample_dataset(df:Dataset, desired_len:int = 100)->Dataset:
    """Oversamples the given dataset for the given label to the desired length. Assumes that the dataset has a column "labels".

    Args:
        dataset (Dataset): Dataset. Must have a column "labels".
        label (str): Label
        desired_len (int, optional): Desired Length. Defaults to 100.

    Returns:
        Dataset: Returns the oversampled dataset
    """
    dfs = []
    for label in set(df["labels"]):
        _df = df.filter(lambda e: e["labels"] == label)
        if len(_df) > desired_len:
            _df = _df.shuffle(seed=42).select(range(desired_len))
        dfs.append(_df)
    
    df = interleave_datasets(dfs, seed=42, stopping_strategy="all_exhausted")

    return df

def encode_one_line(text: str, label: Optional[str] = None)->Tuple[List[str], List[str]]:
    """Encodes one line of text and returns the words and labels. For inference purposes label can be None.

    Args:
        text (str): Text to encode
        label (str, optional): Label. Defaults to None.

    Returns:
        Tuple[List[str], List[str]]: Returns the words and labels
    """

    words = text.split()
    if label is None:
        labels = ["O"]*len(words)
    else:
        labels = [f"B-{label}"] + [f"I-{label}"]*(len(words)-1)
    return words, labels

def prepare_pd_dataset_for_lineclass(df: pd.DataFrame):
    """For Line Label Token classification task, we need to prepare the data
    
    Args:
        df (pd.DataFrame): Dataframe containing the data. Must have columns: rid, text, class_agg. The expected format is ONE report text line per row
        
    Returns:
        pd.DataFrame: Returns the dataframe with the prepared data. Now one row corresponds to the aggregated report text for one report id (rid).
    """

    dict_list = []
    for rid, rid_data in df.groupby("rid"):
        obs_dict = {}
        words, ner_tags, line_label, line_text = [], [], [], []
        for _, row in rid_data.iterrows():
            w, l = encode_one_line(row["text"], row["class_agg"])
            words.extend(w)
            words.append("[BRK]")
            ner_tags.extend(l)
            ner_tags.append("O")
            line_label.append(row["class_agg"])
            line_text.append(row["text"])
        obs_dict["text"] = words
        obs_dict["ner_tags"] = [line_label_token_label2id[l] for l in ner_tags]
        obs_dict["rid"] = rid
        obs_dict["line_label"] = line_label
        obs_dict["line_text"] = line_text
        dict_list.append(obs_dict)
    return pd.DataFrame(data = dict_list)


def information_retrieval(retrieval_model:AutoModelForSequenceClassification,
                          retrieval_tokenizer:AutoTokenizer,
                          text:list[str],
                          label:str)->list[str]:
                          
    """Retrieves relevant information from the given text using the given retrieval model and tokenizer

    Args:
        retrieval_model (AutoModelForSequenceClassification): Retrieval Model
        retrieval_tokenizer (AutoTokenizer): Retrieval Tokenizer
        text (list[str]): Text
        label (str): Label of the relevant information

    Returns:
        list[str]: Returns the relevant text. Output list should have same length as input list.
    """

    print("Retrieving relevant information")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Prepare Data
    max_length = retrieval_tokenizer.model_max_length
    splitted_text = [report.splitlines() for report in text] # Split the text into lines as the retrieval model is trained on line level
    splitted_text = [[line for line in report if line != ""] for report in splitted_text] # Exclude "" lines

    report_lengths = [len(report) for report in splitted_text] # Get the length of each report

    # Flatten the list
    splitted_text = [item for sublist in splitted_text for item in sublist]
    encoded_dataset = retrieval_tokenizer(splitted_text, truncation=True, max_length=max_length, padding = False)
    encoded_dataset = Dataset.from_dict(encoded_dataset)

    # Get DataLoaders
    dataloader = get_DataLoader(encoded_dataset, retrieval_tokenizer, batch_size=32, shuffle=False)

    # Perform Inference
    inference_results = perform_inference(retrieval_model, dataloader, device, output_hidden_states=False)

    # Get the relevant text
    relevant_lines = [splitted_text[i] if line_label_id2label[pred] == label else "" for i, pred in enumerate(inference_results["preds"])]

    # Get back original reports
    relevant_text = []

    for i, report_length in enumerate(report_lengths):
        start_index = sum(report_lengths[:i])
        end_index = start_index + report_length
        relevant_report_text = relevant_lines[start_index:end_index]

        # Remove empty strings
        relevant_report_text = [line for line in relevant_report_text if line != ""]
        relevant_text.append("\n".join(relevant_report_text))

    assert len(relevant_text) == len(text), "Length of relevant text does not match length of input text"

    print("Retrieved relevant information")

    return relevant_text

##########################################################################################################
# Label to id mapping
##########################################################################################################

# MS Label to id
ms_label2id = {'primary_progressive_multiple_sclerosis': 0,
                'relapsing_remitting_multiple_sclerosis': 1,
                'secondary_progressive_multiple_sclerosis': 2,
                'no_ms': 3}

ms_id2label = {v:k for k,v in ms_label2id.items()}

# Line Label to id
line_label_id2label = {0: 'dm',
                       1: 'medo_unk_do_so',
                       2: 'head',
                       3: 'his_sym_cu', 
                       4: 'medms',
                       5: 'labr_labo',
                       6: 'mr',
                       7: 'to_tr',
                       } 

line_label_label2id = {v:k for k,v in line_label_id2label.items()}

line_label_token_label2id = {'B-dm': 0,
 'B-medo_unk_do_so': 1,
 'B-head': 2,
 'B-his_sym_cu': 3,
 'B-medms': 4,
 'B-labr_labo': 5,
 'B-mr': 6,
 'B-to_tr': 7,
 'I-dm': 8,
 'I-medo_unk_do_so': 9,
 'I-head': 10,
 'I-his_sym_cu': 11,
 'I-medms': 12,
 'I-labr_labo': 13,
 'I-mr': 14,
 'I-to_tr': 15,
 'O': 16} 

line_label_token_id2label = {v:k for k,v in line_label_token_label2id.items()}

#########################################################################################################
# Trainer/Training Components
#########################################################################################################

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


def get_optimizer_and_scheduler(model:Union[PeftModel, AutoModelForSequenceClassification],
                                num_training_steps:int, 
                                learning_rate:float)->Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """Returns the optimizer and scheduler for the given model
    
    Args:
        model (PeftModel): Model
        learning_rate (float, optional): Learning Rate. Defaults to LEARNING_RATE.
        
    Returns:
        tuple(torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR): Returns the optimizer and scheduler for the given model
    """

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    return optimizer, lr_scheduler


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


#########################################################################################################
# Inference
#########################################################################################################

def perform_inference(model:Union[PeftModel, AutoModelForSequenceClassification], 
                      dataloader:DataLoader, 
                      device:torch.device,
                      output_hidden_states:bool=False)->dict[str, Union[List[torch.Tensor], List[int]]]:
    """Performs inference on the given dataloader using the given model and returns the last hidden states, labels and predictions.

    Args:
        model (Union[PeftModel, AutoModelForSequenceClassification]): Model to use for inference.
        dataloader (DataLoader): DataLoader to use for inference.
        device (torch.device): Device to use for inference.

    Returns:
        dict[str, Union[List[torch.Tensor], List[int]]]: Returns a dictionary containing the last hidden states List[torch.Tensor], labels List[int]  and predictions List[int].
    """
    model.to(device)
    model.eval()

    with torch.no_grad():
        preds = []
        logits_list = []
        last_hidden_states = []

        for batch in tqdm(dataloader):
            batch.to(device)

            inputs = {k:v for k,v in batch.items() if k != "labels"} # For inference don't need labels. None labels could cause error.
            
            outputs = model(**inputs, output_hidden_states=True)
            logits = outputs.logits.detach().clone().to("cpu")
            logits_list.extend(logits.tolist())
            
            predictions = logits.argmax(dim=-1)
            preds.extend(predictions.tolist())
            
            last_hidden_states.extend(outputs.hidden_states[-1].to("cpu"))

    # For memory efficiency if output_hidden_states is False we don't return the hidden states. Only need them for test evaluation.
    if not output_hidden_states:
        last_hidden_states = []

    return {"last_hidden_state": last_hidden_states, 
            "preds": preds,
            "logits": logits_list,}   


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



########################################################################################################
# Evaluation
########################################################################################################

def plot_embeddings(embeddings: torch.tensor, labels: list[int], title="", method="pca", display_label_mapping: dict = None) -> None:
    """
    Plot embeddings using PCA or UMAP

    Args:
        embeddings (torch.tensor): Embeddings to plot. Shape: (num_samples, embedding_size)
        labels List[int]: Labels in integer format.
        title (str, optional): Title. Defaults to "".
        method (str, optional): Method. Defaults to "pca".
        display_label_mapping (dict, optional): Display label mapping, from int (key) to label (value). Defaults to None.
    """

    # Create a PCA object
    if method == "umap":
        reducer = UMAP()
    elif method == "pca":
        reducer = PCA(n_components=2)
    elif method == "tsne":
        reducer = TSNE(n_components=2, perplexity=5, n_iter=250)
    else:
        raise ValueError("Reducer Method not implemented")

    # Fit and transform the embeddings using the PCA object
    principalComponents = reducer.fit_transform(embeddings)
    print(principalComponents.shape)

    # Create a dataframe with the embeddings and the corresponding labels
    df_embeddings = pd.DataFrame(principalComponents, columns=['x', 'y'])
    df_embeddings['label'] = labels

    # Sort the labels
    df_embeddings = df_embeddings.sort_values(by='label', ascending=True)

    # Display the label mapping
    if display_label_mapping:
        df_embeddings['label'] = df_embeddings['label'].map(display_label_mapping)

    # Plot using Seaborn
    plt.figure(figsize=(8, 6))
    sns.set_theme(style='whitegrid')
    sns.scatterplot(data=df_embeddings, x='x', y='y', hue='label', palette='viridis', alpha=0.7)
    
    # Add a title and legend
    plt.title(title)
    plt.legend(title='Label', loc='upper left')
    
    # Show plot
    plt.tight_layout()
    plt.show()

def performance_metrics(preds:List[int], labels:List[int])->dict[str, float]:

    """
    Returns the accuracy, f1 score, precision and recall

    Args:
        preds (List[int]): Predictions
        labels (List[int]): Labels

    Returns:
        dict[str, float]: Returns the accuracy, f1 score, precision and recall.
    """    

    return {"accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average='macro'),
            "precision": precision_score(labels, preds, average='macro'),
            "recall": recall_score(labels, preds, average='macro')}


def pretty_confusion_matrix(y_true:list[int], y_pred:list[int], label_dict:dict[int, str], title:str="")->None:
    """Plots a pretty confusion matrix using Seaborn's heatmap.

    Args:
        y_true (list[int]): True labels
        y_pred (list[int]): Predicted labels
        label_dict (dict[int, str]): Label dictionary. Maps label indices to display labels. From int2label
        title (str, optional): Title. Defaults to "".

    Returns:
        None
    """

    
    viridis_cmap = plt.get_cmap('viridis_r')
    cmap = sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True)

    # Extract a subset of colors from the "viridis" colormap
    start_index = 120  # Start index of colors to include
    end_index = 200 # End index of colors to include
    subset_colors = viridis_cmap(np.linspace(start_index / 255, end_index / 255, end_index - start_index + 1))

    # Create a custom colormap using the subset of colors
    custom_cmap = mcolors.ListedColormap(subset_colors)
    custom_cmap = sns.color_palette("light:#5A9", as_cmap=True)

    sorted_labels = [item[1] for item in sorted(label_dict.items(), key=lambda x: x[0])]

    # Create the confusion matrix
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)

    # Plotting the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.set_theme(font_scale=1.2)  # Adjust font size for labels
    sns.heatmap(cm, annot=True, fmt='d', cmap=custom_cmap, cbar=False,
                yticklabels=sorted_labels, xticklabels=sorted_labels, alpha=0.9, linewidths=0.5, linecolor='lightgrey')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.yticks(rotation=0)
    plt.grid(False)
    plt.tight_layout()
    plt.title(title)
    plt.show()

def compute_metrics(eval_preds):
    """Computes accuracy, precision, recall and f1 score for the given logits and labels.
    
    Args:
        eval_preds (tuple): Tuple containing logits and labels
        
        Returns:
            dict: Returns the metrics
    """
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def get_df_classificationreport(y_valid, y_pred, labels, param = None, cv_split = None):
    
    # classification report
    dict_cr = classification_report(y_valid, y_pred, output_dict = True)
    
    # data frame of classification report
    df = pd.DataFrame(dict_cr).transpose()
    df.loc['accuracy', ['precision', 'recall']] = np.nan
    df.loc['accuracy', 'support'] = df.loc['macro avg', 'support']
    df = df.astype({'support': int})
    df = df.reset_index().rename(columns = {'index': 'eval_measure'})
    
    # add weights to data frame
    _df = df[df['eval_measure'].isin(labels)]
    weights = [1 if label in ['dm', 'his_sym_cu', 'medms', 'mr'] else 0 for label in sorted(labels)]
    _df = _df.assign(weights = weights)

    # generate data frame for custom weighted values
    list_temp = ['custom_weighted']
    list_temp.append((_df['precision'] * _df['weights']).sum() / _df['weights'].sum())
    list_temp.append((_df['recall'] * _df['weights']).sum() / _df['weights'].sum())
    list_temp.append((_df['f1-score'] * _df['weights']).sum() / _df['weights'].sum())
    list_temp.append(-1)
    df_temp = pd.DataFrame(list_temp, index = df.columns).transpose()

    # add custom weighted values
    df = pd.concat([df, df_temp], axis = 0, ignore_index = True)
    df = df.astype({'precision': float, 'recall': float, 'f1-score': float, 'support': int})
    
    # assign grid search parameters
    if param is not None:
        for key, value in param.items():
            df = df.assign(temp = value)
            df = df.rename(columns = {'temp': key})
    
    # assign cv split number
    if cv_split is not None:
        df = df.assign(cv_split = cv_split)

    return df            

def majority_vote(current_predictions: list)->str:
    """Helper function to perform majority vote on the given predictions in a line.
    
    Args:
        current_predictions (list): List of predictions for the current line
        
    Returns:
        str: Returns the majority vote prediction
            
    """
    # Get the most common prediction
    remapped_predictions = [line_label_token_id2label[p][2:] for p in current_predictions] # Remove the B- or I- prefix

    # Counte the occurrences of each class
    class_counts = Counter(remapped_predictions)

    # Find the maximum count
    max_count = max(class_counts.values())

    # Find all classes with the maximum count
    most_common_predictions = [prediction for prediction, count in class_counts.items() if count == max_count]

    # Return the first one among tied classes
    return most_common_predictions[0]
    

def group_labels_by_text(tokenized_texts:list, predictions:list):
    """Groups the labels by rid text and returns the line predictions. !!! Assumes that the tokenized_texts and predictions are sorted by rid !!!
    !!!If there are reports longer than the max length of the model, the labels for the tokens after the max length will be discarded!!!
    This means evaluation is only done on the first max length tokens of the report.
    
    Args:
        tokenized_texts (list): List of tokenized texts
        predictions (list): List of predictions. One prediction per token in tokenized_texts.
        
    Returns:
        list: Returns one prediction per line. Aggregated through majority vote.
            
    """
    line_predictions = []
    current_text_predictions = []

    for token, prediction in zip(tokenized_texts, predictions):
        # The zip will make sure that padded elements from predictions are ignored as overflow is discarded
        # Check for the end of a text line (using the BRK token)
        if '[BRK]' == token:
            # Don't need to add BRK prediction as it is not a token, at this point just add the current text predictions
            line_prediction = majority_vote(current_text_predictions)
            line_predictions.append(line_prediction)
            current_text_predictions = []
            
        else:
            current_text_predictions.append(prediction)

    return line_predictions


def get_results_from_token_preds(predictions:np.ndarray,
                                 dataset:DatasetDict,
                                 tokenizer:AutoTokenizer,
                                 split:str="test"):
    """Get a list of line labels from the token predictions. This is done by finding the line breaks in the text
    for each rid, then take a majority vote of the labels for each line. All the line labels are concatenated
    to a list that should match the labels in the dataset. Because of truncation might have bugs
    
    Args:
        predictions (np.ndarray): shape (n_samples, max_len, n_labels)
        dataset (DatasetDict): must contain "input_ids" and "line_label" for the specified split. Line label is a list of one label per line.
        tokenizer (AutoTokenizer): Tokenizer. Needed to convert the input_ids to tokens and match lines.
        split (str, optional): Split that was used to calculate predictions Defaults to "test"."""
    
    predictions = np.argmax(predictions, axis=2)

    print(f"Predictions shape: {predictions.shape}")
    
    data = []
    # preds, labs, rid, text = [], [], [], []
    for i in range(len(dataset[split])):
        data_dict = {}
        # Because of truncation only add labels up to the max length
        recoded_preds = group_labels_by_text(tokenizer.convert_ids_to_tokens(dataset[split][i]["input_ids"]), predictions[i,:])
        max_len = len(recoded_preds)
        data_dict["preds"] = recoded_preds
        # preds.append(recoded_preds)
        data_dict["labs"] = dataset[split][i]["line_label"][:max_len]
        # labs.append(dataset[split][i]["line_label"][:max_len])
        data_dict["rid"] = dataset[split][i]["rid"]
        # rid.append(dataset[split][i]["rid"])
        data_dict["text"] = dataset[split][i]["line_text"][:max_len]
        # text.append(dataset[split][i]["line_text"][:max_len])
        data.append(data_dict)
        

    # return preds, labs, rid, text
    return data

########################################################################################################
# Prompting
########################################################################################################


def get_format_fun(prompting_strategy:str)->Callable[[str],str]:
    """Get format function for prompting strategy

    Args:
        prompting_strategy (str): prompting strategy. Must be one of zero_shot_vanilla, zero_shot_instruction, few_shot_vanilla, few_shot_instruction, two_steps, or all. Defaults to zero_shot_vanilla.

    Returns:
        Callable[[str],str]: format function

    """
    if prompting_strategy == "zero_shot_vanilla":
        return zero_shot_base

    elif prompting_strategy == "zero_shot_instruction":
        return zero_shot_instruction

    elif prompting_strategy == "few_shot_vanilla":
        return few_shot_base

    elif prompting_strategy == "few_shot_instruction":
        return few_shot_instruction

    elif prompting_strategy == "two_steps":
        return two_steps_one

    else:
        raise ValueError(f"prompting_strategy must be one of zero_shot_vanilla, zero_shot_instruction, few_shot_vanilla, few_shot_instruction, two_steps, or all. Got {prompting_strategy}")
    
                    
def zero_shot_base(input:str, system_prompt:str, task_instruction:str, *args, **kwargs)->str:
    """Zero-shot base for Llama prompting

    Args:
        input (str): user input (medical report)
        system_prompt (str): system prompt
        task_instruction (str): instruction for the task

    Returns:
        str: reformatted medical report

    """
    base_prompt = "[INST]<<SYS>>{system_prompt}<</SYS>>\n\n{task_instruction}\nHere is the Input:\n{input}[/INST]"
    
    input = base_prompt.format(system_prompt = system_prompt, task_instruction = task_instruction, input =  input)

    return input
    
def zero_shot_instruction(input:str, system_prompt:str, task_instruction:str, *args, **kwargs)->str:
    """Zero-shot instruction for the MS extraction task
    
    Args:
        input (str): user input (medical report)
        system_prompt (str): system prompt
        task_instruction (str): instruction for the task
        
        Returns:
            str: reformatted medical report with instruction
            
    """

    instruction_base_prompt = "[INST]<<SYS>>{system_prompt}<</SYS>>\n### Instruction:\n{task_instruction}\n\n### Input:\n{input}[/INST]\n\n### Output:\n"
    input = instruction_base_prompt.format(system_prompt = system_prompt, task_instruction = task_instruction, input = input)

    return input



def few_shot_base(input:str, system_prompt:str, task_instruction:str, examples:list[dict], *args, **kwargs)->str:
    """Few Shot base for the MS extraction task

    Args:
        input (str): user input (medical report)
        system_prompt (str): system prompt
        task_instruction (str): instruction for the task
        examples (list[dict]): examples for the task. Each dict contains a text and a label.

    Returns:
        str: reformatted medical report with examples

    """
    base_prompt = ("[INST]<<SYS>>{system_prompt}<</SYS>>\n\n{task_instruction}\n"
                   "Here are some examples to help you understand the task:\n\n{examples} \n"
                   "Please provide your answer for the following Input.\n"
                    "Input:\n{input}[/INST]"
                   )
    
    insert_examples = ""
        
    for example in examples:
        text = example["text"]
        label = example["labels"]
        insert_examples += f"Input:\n{text}\nOutput:\n{label}\n\n"
    
    input = base_prompt.format(system_prompt = system_prompt, task_instruction = task_instruction, examples = insert_examples, input = input)
    return input


def few_shot_instruction(input:str, system_prompt:str, task_instruction:str, examples:list[dict], *args, **kwargs)->str:
    """Few Shot Instruction for the MS extraction task

    Args:
        input (str): user input (medical report)
        system_prompt (str): system prompt
        task_instruction (str): instruction for the task
        examples (list[dict]): examples for the task. Each dict contains a text and a label.

    Returns:
        str: reformatted medical report with examples and instruction

    """
    base_prompt = ("[INST]<<SYS>>{system_prompt}<</SYS>>\n\n### Instruction:\n{task_instruction}\n"
                   "Here are some examples to help you understand the task:\n{examples}\n"
                   "Please provide your answer for the following Input.\n\n"
                   "### Input:\n{input}[/INST]\n\n### Output:\n")

    insert_examples = ""

    for example in examples:
        text = example["text"]
        label = example["labels"]
        insert_examples += f"Input:\n{text}\nOutput:\n{label}\n\n"
    
    input = base_prompt.format(system_prompt = system_prompt, task_instruction = task_instruction, examples = insert_examples, input = input)

    return input


def two_steps_one(input: str, *args, **kwargs)->str:
    """Two Steps One for the MS extraction task. Encodes the report for first turn of the dialogue.

    Args:
        input (str): medical report

    Returns:
        str: reformatted medical report with base

    """
    base_prompt = "<s>[INST]<<SYS>>{system_prompt}<</SYS>>\n\n{instruction}{input}[/INST]"
    system_prompt =  ("\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
                      "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
                       "Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make "
                        "any sense, or is not factually coherent, explain why instead of answering something not correct. If you don’t "
                        "know the answer to a question, please don’t share false information.\n"
                        )
    instruction = ("Your task is to extract relevant information about the multiple sclerosis diagnosis from the provided German medical report. "
                   "Identify and summarize all sections discussing \"Multiple Sklerose\" paying attention to the exact type of multiple sclerosis. "
                   "There are three types:\n"
                   "primär progrediente Multiple Sklerose (PPMS)\n"
                   "sekundär progrediente Multiple Sklerose (SPMS)\n"
                   "schubförmige Multiple Sklerose (RRMS)\n"
                   "If the report lacks information about multiple sclerosis, respond with \"not enough info\". "
                   "\nHere is the Medical Report:\n "
                   )
                   
    input = base_prompt.format(system_prompt = system_prompt, instruction = instruction, input =  input)
    return input


def format_prompt(text: list[str], format_fun: Callable[[List[str]], List[str]], *args, **kwargs) -> list[str]:
    """
    Formats a list of texts using a given formatting function. Used for formatting text with a prompt template.

    Args:
        text (list[str]): list of strings to be formatted
        format_fun (Callable[list[str], list[str]]): formatting function. Specify additional arguments using *args and **kwargs.

    Returns:
        list[str]: list of formatted strings
    """
    return [format_fun(t, **kwargs) for t in text]

def outlines_prompting(text: list[str], generator: SequenceGenerator, batch_size: int = 4)-> list[Union[str, BaseModel]]:
    """
    Generates a list of sequences using the given outlines generator and sampler.

    Args:
        text (list[str]): list of strings to be used as prompts
        generator (outlines.SequenceGenerator): outlines generator
        batch_size (int, optional): batch size. Defaults to 4.

    Returns:
        list[Union[str, pydantic.BaseModel]]: list of generated sequences
    """
    dataloader = DataLoader(text, batch_size = batch_size, shuffle = False)

    results = []

    for batch in tqdm(dataloader):
        answer = generator(batch)
        results.extend(answer)

    return results

def get_sampler(sampler_name:str)->samplers.Sampler:
    """Get sampler for the outlines generator

    Args:
        sampler_name (str): sampler name. Must be one of greedy, multinomial or beam.

    Returns:
        samplers.Sampler: sampler
    """

    if sampler_name=="greedy":
        sampler = samplers.greedy()
    elif sampler_name=="multinomial":
        sampler = samplers.multinomial()
    elif sampler_name=="beam":
        sampler = samplers.beam_search(beams=2)
    else:
        raise ValueError(f"Invalid sampler. Must be one of greedy, multinomial or beam. Got {sampler_name}")
    
    return sampler
  
def get_outlines_generator(model: Callable, sampler: outlines.samplers.Sampler, task: str = "text", *args, **kwargs) -> outlines.generate.SequenceGenerator:
    if task == "choice":
        choices = kwargs.get("choices")
        assert choices is not None, "For 'choice' generation, you need to specify a list of choices as 'choices' argument."
        assert isinstance(choices, list), "The 'choices' argument must be a list of strings."
        return outlines.generate.choice(model, choices=choices, sampler=sampler)
    
    elif task == "json":
        schema = kwargs.get("schema")
        assert schema is not None, "For 'json' generation, you need to specify a Pydantic Base model as 'schema' argument."
        assert issubclass(schema, BaseModel), "The 'schema' argument must be a subclass of the Pydantic Base model."
        return outlines.generate.json(model, schema, sampler=sampler)
    
    elif task == "text":
        return outlines.generate.text(model, sampler=sampler)
    
    else:
        raise f"Task type {task} not implemented"

def get_pydantic_schema(schema_name: str)->BaseModel:
    if schema_name == "medication":
        class MedicationUnit(str, Enum):
            mg = "mg"
            ug = "ug"
            g = "g"
            stk = "stk"
            tropfen = "tropfen"
            ml = "ml"
            unknown = "unknown"

        class Medication(BaseModel):
            name: str
            unit: MedicationUnit
            amount: float
            morning: float
            noon: float
            evening: float
            night: float

        class MedicationList(BaseModel):
            medications: list[Medication]

        return MedicationList
    else:
        raise f"Schema {schema_name} not implemented"


