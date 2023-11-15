import matplotlib.pyplot as plt
import torch
import polars as pl
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from datasets import DatasetDict
import umap
import numpy as np

def plot_embeddings(embeddings: torch.tensor, labels:pl.Series, title = "plot", method="pca"):

    # Create a PCA object
    if method == "umap":
        reducer = umap.UMAP()
    elif method == "pca":
        reducer = PCA(n_components=2)
    elif method == "tsne":
        reducer = TSNE(n_components=2, perplexity=5, n_iter=250)

    # Fit and transform the embeddings using the PCA object
    principalComponents = reducer.fit_transform(embeddings)

    # Create a dataframe with the embeddings and the corresponding labels
    df_embeddings = pl.DataFrame(principalComponents)
    df_embeddings.columns = ['x', 'y']
    df_embeddings= df_embeddings.with_columns(
        pl.lit(labels).alias("label")
        .cast(pl.Categorical)
        )
    
    for label in df_embeddings['label'].unique():
        _df = df_embeddings.filter(df_embeddings['label'] == label)
        plt.scatter(_df['x'], _df['y'], alpha=0.5)
        plt.legend(df_embeddings['label'].unique())

    # Add a title and show the plot
    plt.title(title)

    plt.show()
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