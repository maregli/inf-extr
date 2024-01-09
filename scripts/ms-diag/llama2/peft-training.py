from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, BitsAndBytesConfig, get_linear_schedule_with_warmup

from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict

from peft import get_peft_config, get_peft_model, prepare_model_for_kbit_training, PeftConfig, PromptEncoderConfig, LoraConfig, PeftModel, PromptTuningConfig, PromptTuningInit

import torch
from torch.utils.data import DataLoader


import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src import paths

from sklearn.metrics import f1_score

import pandas as pd

from tqdm import tqdm

import argparse
from typing import Tuple

MODEL_NAME = 'llama2'
QUANTIZATION = "4bit"

TRUNCATION_SIZE = 256

BATCH_SIZE = 4
LEARNING_RATE = 1e-3
NUM_EPOCHS = 4

GRADIENT_ACCUMULATION_STEPS = None

PEFT_TYPE = "lora"

DATA = "original"

CLASSIFICATION = "multi-class"



def parse_args():
    parser = argparse.ArgumentParser(description="Zero Shot Classification with Llama2-Chat")
    parser.add_argument("--job_id", type=str, default="unknown", help="Job ID")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME, help="Name of model to be used. Defaults to llama2. Must be saved in the path: paths.MODEL_PATH/model_name")
    parser.add_argument("--quantization", type=str, default=QUANTIZATION, help="Quantization. Must be one of 4bit, bfloat16 or float16. Defaults to 4bit")
    parser.add_argument("--truncation_size", type=int, default=TRUNCATION_SIZE, help="Truncation Size of the input text. Defaults to 256")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch Size. Defaults to 4")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning Rate. Defaults to 1e-3")
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS, help="Number of Epochs. Defaults to 4")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=GRADIENT_ACCUMULATION_STEPS, help="Gradient Accumulation Steps. If not set, it will be set to 8 // batch_size or 1 if batch_size >= 8")
    parser.add_argument("--peft_type", type=str, default=PEFT_TYPE, help="PEFT Type. Must be one of lora, prefix, ptune or prompt. Defaults to lora")
    parser.add_argument("--data", type=str, default=DATA, help="Data. Must be one of original, zero-shot or augmented. Defaults to original")
    parser.add_argument("--classification", type=str, default=CLASSIFICATION, help="Classification. Must be one of multi-class or binary. Defaults to multi-class")

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

def load_model_and_tokenizer(model_name:str=MODEL_NAME, quantization:str = QUANTIZATION, classification:str=CLASSIFICATION)->Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Loads the model and tokenizer from the given path and returns the compiled model and tokenizer.
    
    Args:
        model (str): Model Name. Assumes that the model is saved in the path: paths.MODEL_PATH/model. Defaults to MODEL.
        quantization (str, optional): Quantization. Must be one of 4bit or bfloat16. Defaults to QUANTIZATION.
        classification (str, optional): Classification. Must be one of multi-class or binary. Defaults to CLASSIFICATION.

        Returns:
            tuple(AutoModelForCausalLM, AutoTokenizer): Returns the compiled model and tokenizer
            
    """

    # Number of labels
    if classification == "multi-class":
        num_labels = 3
    elif classification == "binary":
        num_labels = 2
    else:
        raise ValueError("Classification must be one of multi-class or binary")
    
    ### Model
    if quantization == "bfloat16":
        model = AutoModelForSequenceClassification.from_pretrained(paths.MODEL_PATH/model_name, 
                                                                   torch_dtype=torch.bfloat16,
                                                                   num_labels = num_labels)
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
                                                                    #device_map="auto"
                                                                   quantization_config=bnb_config,
                                                                   num_labels = num_labels)
        model = prepare_model_for_kbit_training(model)
    else:
        raise ValueError("Quantization must be one of 4bit or bfloat16")
    
    model.gradient_checkpointing_enable()
    
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


def load_data(data:str=DATA)->DatasetDict:
    """Loads the data for MS-Diag task and returns the dataset dictionary

    Args:
        data (str, optional): Data. Must be one of original, zero-shot or augmented. Defaults to DATA.
    
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
            
    return df

def prepare_data(df:DatasetDict, tokenizer:AutoTokenizer, peft_config:PeftConfig,truncation_size:int = TRUNCATION_SIZE)->Dataset:

    # Label to id
    label2id = {'primary_progressive_multiple_sclerosis': 0,
                'relapsing_remitting_multiple_sclerosis': 1,
                'secondary_progressive_multiple_sclerosis': 2}
    id2label = {v:k for k,v in label2id.items()}
    
    # For Prompt Tuning, we need to add the prefix to the input text
    if peft_config.is_prompt_learning:
        max_length = tokenizer.model_max_length - peft_config.num_virtual_tokens
    else:
        max_length = tokenizer.model_max_length
        
    truncation_size = min(truncation_size, max_length)

    def tokenize_function(examples):
        outputs = tokenizer(examples["text"], truncation=True, max_length=truncation_size)
        outputs["labels"] = [label2id[label] for label in examples["labels"]]
        return outputs

    encoded_dataset = df.map(tokenize_function, batched=True, remove_columns=["text", "rid", "date"])

    return encoded_dataset

def select_peft_config(model:AutoModelForSequenceClassification, peft_type:str = PEFT_TYPE)->PeftConfig:
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
        peft_config = get_peft_config(config)
    
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

def prepare_peft_model(model:AutoModelForSequenceClassification, peft_config:PeftConfig)->PeftModel:
    """Prepares the model for PEFT training and returns the model
    
    Args:
        model (AutoModelForSequenceClassification): Base Model.
        peft_config (PeftConfig): PEFT Config    
    Returns:
        PeftModel: Returns the peft model
    """

    # Get the PEFT model
    peft_model = get_peft_model(model, peft_config)

    # Print trainable parameters
    peft_model.print_trainable_parameters()

    return peft_model

def get_DataLoader(df:Dataset, tokenizer:AutoTokenizer, batch_size:int = BATCH_SIZE, shuffle:bool = True)->DataLoader:
    """Returns a DataLoader for the given dataset dictionary
    
    Args:
        df (Dataset): HF Dataset
        tokenizer (AutoTokenizer): Tokenizer
        batch_size (int, optional): Batch size. Defaults to BATCH_SIZE.
        shuffle (bool, optional): Shuffle. Defaults to False.
        
    Returns:
        DataLoader: Returns a DataLoader for the given dataset dictionary
    """

    # Default collate function 
    collate_fn = DataCollatorWithPadding(tokenizer, padding="longest", pad_to_multiple_of=8)

    dataloader = torch.utils.data.DataLoader(dataset=df, collate_fn=collate_fn, batch_size=batch_size, shuffle = shuffle) 

    return dataloader

def get_optimizer_and_scheduler(model:PeftModel,
                                num_training_steps:int, 
                                learning_rate:float = LEARNING_RATE)->Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
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

def train_loop(peft_model:PeftModel, 
               train_dataloader:DataLoader, 
               eval_dataloader:DataLoader, 
               device:torch.device,
               peft_model_name:str,
               num_epochs:int = NUM_EPOCHS, 
               learning_rate:float = LEARNING_RATE,
               gradient_accumulation_steps:int = GRADIENT_ACCUMULATION_STEPS,
               )->None:
    # Seed
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Optimizer and Scheduler
    num_training_steps = num_epochs * len(train_dataloader)
    optimizer, lr_scheduler = get_optimizer_and_scheduler(peft_model, num_training_steps)

    # Training
    peft_model.to(device)

    for epoch in range(num_epochs):
        peft_model.train()
        total_loss = 0
        bar = tqdm(train_dataloader)

        for step, batch in enumerate(bar):
            optimizer.zero_grad()
            batch.to(device)
            outputs = peft_model(**batch)
            
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
            bar.set_description(f"Epoch: {epoch}, Loss: {loss.item():.4f}")

        peft_model.eval()

        with torch.no_grad():
            eval_loss = 0
            eval_preds = []
            labels = []
            for step, batch in enumerate(tqdm(eval_dataloader)):
                batch.to(device)
                outputs = peft_model(**batch)
                    
                predictions = outputs.logits.argmax(dim=-1)
                eval_preds.extend(predictions.tolist())
                labels.extend(batch['labels'].tolist())
                
                loss = outputs.loss
                eval_loss += loss.detach().float()
                
        f1 = f1_score(labels, eval_preds, average='macro')
        
        if epoch == 0:
            max_f1 = 0
            min_eval_loss = eval_loss
            print(f"Saving Model at {paths.MODEL_PATH/peft_model_name}")
            peft_model.save_pretrained(paths.MODEL_PATH/peft_model_name)

        if f1 > max_f1:
            max_f1 = f1
            min_eval_loss = eval_loss
            print(f"Saving Model at {paths.MODEL_PATH/peft_model_name}")
            peft_model.save_pretrained(paths.MODEL_PATH/peft_model_name)

        elif f1 == max_f1 and eval_loss < min_eval_loss:
            min_eval_loss = eval_loss
            print(f"Saving Model at {paths.MODEL_PATH/peft_model_name}")
            peft_model.save_pretrained(paths.MODEL_PATH/peft_model_name)

        eval_epoch_loss = eval_loss / len(eval_dataloader)
        train_epoch_loss = total_loss / len(train_dataloader)
        print(f"{epoch=}: {train_epoch_loss=} {eval_epoch_loss=} {f1=}")

def main():

    args = parse_args()
    JOB_ID = args.job_id
    MODEL_NAME = args.model_name
    QUANTIZATION = args.quantization
    TRUNCATION_SIZE = args.truncation_size
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    NUM_EPOCHS = args.num_epochs
    GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation_steps
    PEFT_TYPE = args.peft_type
    DATA = args.data
    CLASSIFICATION = args.classification

    if GRADIENT_ACCUMULATION_STEPS is None:
        if BATCH_SIZE >= 8:
            GRADIENT_ACCUMULATION_STEPS = 1
        else:
            GRADIENT_ACCUMULATION_STEPS = 8 // BATCH_SIZE

    # Name Model for saving
    peft_model_name = f"ms-diag_{MODEL_NAME}_{QUANTIZATION}_{PEFT_TYPE}_{DATA}_{TRUNCATION_SIZE}"
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Check GPU Memory
    check_gpu_memory()

    # Load Model and Tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name=MODEL_NAME, quantization=QUANTIZATION, classification=CLASSIFICATION)

    print("Loaded Model and Tokenizer")

    # Get Peft Model
    peft_config = select_peft_config(model=model, peft_type=PEFT_TYPE)
    peft_model = prepare_peft_model(model=model, peft_config=peft_config)

    print("Loaded PEFT Model")

    # Load Data
    df = load_data(data=DATA)

    # Prepare Data
    encoded_dataset = prepare_data(df, tokenizer, peft_config, truncation_size=TRUNCATION_SIZE)

    print("Loaded Data")

    # Get DataLoaders
    train_dataloader = get_DataLoader(encoded_dataset["train"], tokenizer, batch_size=BATCH_SIZE)
    eval_dataloader = get_DataLoader(encoded_dataset["validation"], tokenizer, batch_size=BATCH_SIZE, shuffle=False)

    # Train Loop
    print("Starting Training")
    train_loop(peft_model=peft_model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                device=device,
                num_epochs=NUM_EPOCHS,
                learning_rate=LEARNING_RATE,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                peft_model_name=peft_model_name,
                )
    print("Training Finished")
    return

if __name__ == "__main__":
    main()


