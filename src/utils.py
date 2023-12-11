from typing import List, Dict, Tuple, Union, Optional

import torch
from torch.utils.data import DataLoader

from sklearn.preprocessing import OneHotEncoder

from datasets import DatasetDict
from datasets import Dataset

import numpy as np

from transformers import AutoModelForSequenceClassification

import tqdm

class MedDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe: DatasetDict, tokenizer, max_length: int = 512, split: str = 'train'):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(np.stack(self.dataframe['train']['label']).reshape(-1, 1))
        self.labels = self.enc.transform(np.stack(self.dataframe[split]['label']).reshape(-1, 1))
        self.encodings = self.tokenizer(self.dataframe[split]['text'], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')

    def __getitem__(self, idx):
        item = {key: (val[idx].clone().detach()) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
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
    for batch in tqdm.tqdm(dataloader):
        input_ids = torch.stack(batch['input_ids'], dim=1).to(device)
        attention_mask = torch.stack(batch['attention_mask'], dim=1).to(device)
        token_type_ids = torch.stack(batch['token_type_ids'], dim=1).to(device)
        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
            embeddings.append(output.hidden_states[-1].cpu())
            logits.append(output.logits.cpu())
            labels.append(torch.stack(batch['labels'], dim = 1))
    return {"embeddings": torch.cat(embeddings, dim=0), "logits": torch.cat(logits, dim=0), "labels": torch.cat(labels, dim = 0)}

def train_transformer(
        number: torch.Tensor
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Train transformer on number
    
    Args:
        number (torch.Tensor): Number to train on

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]: Tuple of (input_ids, attention_mask, token_type_ids)
    """
    return number, None, None