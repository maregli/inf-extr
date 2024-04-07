# Utilizing Deep Language Models for the Medical Information Extraction

## Overview

The emergence of Large Language Models (LLMs) has brought about significant innovations across various sectors, including the field of healthcare informatics.  This thesis explores the application of LLMs for extracting structured information from German medical reports, a task that is particularly challenging for the clinical domain given the specialized vocabulary, the lack of high-quality, annotated datasets, and the prohibitive costs associated with manual data annotation. Our approach utilized prompt templates, few-shot learning, and structured text generation to steer the unrestricted text outputs of LLMs. This ranged from more straightforward tasks such as extracting a single diagnosis to intricate relation extraction tasks, including identifying multiple medications and their specific attributes. By employing LLMs such as Llama2-MedTuned, our findings reveal that LLMs are competitive with previous non-LLM approaches of the Swiss Data Science Center (SDSC), sometimes even outperforming them, all while requiring minimal to no training data.

This GitHub Repo contains all the code for out analysis. Because of data privacy concerns, the data is only available on the secure leomed cluster.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data](#data)
- [Running the Project](#running-the-project)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Step 1: Install Poetry

Ensure [Poetry](https://python-poetry.org/docs/) is installed on your system. Poetry manages project dependencies and environments.

### Step 2: Install Dependencies

Clone the repository and navigate to the project directory. Use the provided `setup.sh` script to install all necessary dependencies.

```bash
git clone https://github.com/maregli/inf-extr.git
cd inf-extr
./setup.sh
```

## Project Structure

Below is the file structure of our project, which includes notebooks for data processing and evaluations, scripts for various processing tasks, source code, and thesis-related images and results.

```bash
├── README.md
├── notebooks
│   ├── data_preprocessing
│   ├── line-label_prediction
│   ├── llama2-finetune
│   ├── medication
│   ├── ms-prediction
│   └── side-effects
├── poetry.lock
├── pyproject.toml
├── scripts
│   ├── line-label
│   ├── llama2-finetune
│   ├── medication
│   ├── ms-diag
│   └── side-effects
├── setup.sh
├── src
│   ├── __init__.py
│   ├── __pycache__
│   ├── data
│   ├── paths.py
│   └── utils.py
└── thesis
```

### Key Directories and Files

- **notebooks/**: Contains Jupyter notebooks for data preprocessing as well as evaluations of the different tasks (Content classification, diagnosis extraction, medication extraction, side effects extraction, fine-tuning)
- **scripts/**: Includes scripts for training the models for the different tasks and performing inference.
- **src/**: Source code for the project, including utility functions and data management.
- **thesis/**: Contains images and CSV files with results related to the thesis, including graphs, embeddings, and performance metrics.
- **pyproject.toml & poetry.lock**: Configuration files for project dependencies.
- **setup.sh**: A script for setting up the project environment.

For more details on each directory and its contents, refer to the individual files and folders within the project.

## Data and Resources

The data, resources and results used for this thesis can be found on the leomed cluster under:

```
/cluster/dataset/midatams/inf-extr/
```

Make sure that the import paths, especially the project root in ```src/paths.py``` matches the data path.

## Running Project

- Preprocessing: Begin with the notebooks in `notebooks/data_preprocessing` for initial data handling.
- Training: Use scripts in `scripts/` for model training. Check each script for command line options.
- Inference: For model inference, refer to the relevant scripts in `scripts/`.
- Evaluation: Evaluation notebooks are located in `notebooks/`, organized by task.