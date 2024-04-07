# Utilizing Deep Language Models for the Medical Information Extraction

This is for my master thesis.

I updated it now.

I update it again.

To install dependencies:
poetry install 
then to build flash-attention: FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE poetry run python -m pip install flash-attn --no-build-isolation
maybe you can also try without 

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
    ├── line-label-embeddings.png
    ├── line-label_line_cm.png
    ├── line-label_line_embeddings.png
    ├── line-label_line_results.csv
    ├── line-label_token_cm.png
    ├── line-label_token_embeddings.png
    ├── line-label_token_results.csv
    ├── line-level-cm.png
    ├── llama2-finetuning-1024.png
    ├── llama2-finetuning-512.png
    ├── medication_results_13b.csv
    ├── medication_results_7b.csv
    ├── medication_results_rule.csv
    ├── ms_diag_medbert_cm_base.png
    ├── ms_diag_medbert_cm_s2a.png
    ├── ms_diag_medbert_embeddings_base.png
    ├── ms_diag_medbert_embeddings_s2a.png
    ├── ms_pred_results_prompt13b.csv
    ├── ms_pred_results_prompt13b_no.csv
    ├── ms_pred_results_prompt7b.csv
    ├── roc_curve.png
    ├── se_lora_f1.csv
    ├── se_lora_precision.csv
    ├── se_lora_recall.csv
    ├── se_prc.png
    ├── se_roc.png
    ├── se_thesis7b_f1.csv
    ├── se_thesis7b_precision.csv
    ├── se_thesis7b_recall.csv
    ├── se_threshold_f1.png
    ├── se_threshold_precision.png
    ├── se_threshold_recall.png
    ├── token-label-embeddings.png
    └── token-level-cm.png