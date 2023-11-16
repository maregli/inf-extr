import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from datasets import DatasetDict
import umap
import numpy as np

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