import matplotlib.pyplot as plt
import torch
import polars as pl
from sklearn.decomposition import PCA
import umap

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