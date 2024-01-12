from typing import List, Dict, Tuple, Union, Optional

import torch
from torch.utils.data import DataLoader

from peft import prepare_model_for_kbit_training, PeftConfig, PeftModel

from datasets import DatasetDict, Dataset, load_dataset, concatenate_datasets

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, BitsAndBytesConfig, get_linear_schedule_with_warmup

from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, ConfusionMatrixDisplay

from umap import UMAP

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
    # If model is bert model, use bert tokenizer
    if "bert" in model_name:
        padding_side = "right"
    else:
        padding_side = "left"

    tokenizer = AutoTokenizer.from_pretrained(paths.MODEL_PATH/model_name, padding_side=padding_side)

    # Check if the pad token is already in the tokenizer vocabulary
    if tokenizer.pad_token_id is None:
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
    model.to(device)
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


def train_loop(model:Union[PeftModel, AutoModelForSequenceClassification], 
               train_dataloader:DataLoader, 
               eval_dataloader:DataLoader, 
               device:torch.device,
               finetuned_model_name:str,
               num_epochs:int = 4, 
               learning_rate:float = 1e-03,
               gradient_accumulation_steps:int = 1,
               )->None:
    
    """Trains the given model using the given dataloader and returns the trained model.

    Args:
        model (Union[PeftModel, AutoModelForSequenceClassification]): Model to train.
        train_dataloader (DataLoader): Train DataLoader.
        eval_dataloader (DataLoader): Eval DataLoader.
        device (torch.device): Device to use for training.
        finetuned_model_name (str): Name under which finetuned model is saved.
        num_epochs (int, optional): Number of epochs. Defaults to 4.
        learning_rate (float, optional): Learning Rate. Defaults to 1e-03.
        gradient_accumulation_steps (int, optional): Gradient Accumulation Steps. Defaults to 1.

    Returns:
        None: Trains the given model using the given dataloader and returns the trained model.
    """

    # Seed
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Optimizer and Scheduler
    num_training_steps = num_epochs * len(train_dataloader)
    optimizer, lr_scheduler = get_optimizer_and_scheduler(model, num_training_steps, learning_rate)

    # Training
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        bar = tqdm(train_dataloader)

        for step, batch in enumerate(bar):
            optimizer.zero_grad()
            batch.to(device)
            outputs = model(**batch)
            
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
            bar.set_description(f"Epoch: {epoch}, Loss: {loss.item():.4f}")

        model.eval()

        with torch.no_grad():
            eval_loss = 0
            eval_preds = []
            labels = []
            for step, batch in enumerate(tqdm(eval_dataloader)):
                batch.to(device)
                outputs = model(**batch)
                    
                predictions = outputs.logits.argmax(dim=-1)
                eval_preds.extend(predictions.tolist())
                labels.extend(batch['labels'].tolist())
                
                loss = outputs.loss
                eval_loss += loss.detach().float()
                
        f1 = f1_score(labels, eval_preds, average='macro')
        
        if epoch == 0:
            max_f1 = 0
            min_eval_loss = eval_loss
            print(f"Saving Model at {paths.MODEL_PATH/finetuned_model_name}")
            model.save_pretrained(paths.MODEL_PATH/finetuned_model_name)

        if f1 > max_f1:
            max_f1 = f1
            min_eval_loss = eval_loss
            print(f"Saving Model at {paths.MODEL_PATH/finetuned_model_name}")
            model.save_pretrained(paths.MODEL_PATH/finetuned_model_name)

        elif f1 == max_f1 and eval_loss < min_eval_loss:
            min_eval_loss = eval_loss
            print(f"Saving Model at {paths.MODEL_PATH/finetuned_model_name}")
            model.save_pretrained(paths.MODEL_PATH/finetuned_model_name)

        eval_epoch_loss = eval_loss / len(eval_dataloader)
        train_epoch_loss = total_loss / len(train_dataloader)
        print(f"{epoch=}: {train_epoch_loss=} {eval_epoch_loss=} {f1=}")

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

def plot_embeddings(embeddings: torch.tensor, labels:List[int], title = "plot", method="pca")->None:
    """
    Plot embeddings using PCA or UMAP

    Args:
        embeddings (torch.tensor): Embeddings to plot. Shape: (num_samples, embedding_size)
        labels List[int]: Labels in integer format.
        title (str, optional): Title. Defaults to "plot".
        method (str, optional): Method. Defaults to "pca".
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
    
    for label in df_embeddings['label'].unique():
        _df = df_embeddings[df_embeddings['label'] == label]
        plt.scatter(_df['x'], _df['y'], alpha=0.5)
        plt.legend(df_embeddings['label'].unique())

    # Add a title and show the plot
    plt.title(title)

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

def plot_confusion_matrix(preds:List[int], labels:List[int], title:str = "Confusion Matrix", label2id:dict = None)->None:
    """
    Plots the confusion matrix

    Args:
        preds (List[int]): Predictions
        labels (List[int]): Labels
        title (str, optional): Title. Defaults to "Confusion Matrix".
        id2label (dict, optional): Id to label mapping. Defaults to None.
    """    
    ConfusionMatrixDisplay.from_predictions(y_true = labels, y_pred = preds, display_labels=label2id, xticks_rotation="vertical")
    plt.title(title)