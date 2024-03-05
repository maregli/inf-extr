
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src import paths

from src.utils import load_model_and_tokenizer
from peft import PeftModel

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Llama2-finetuning unsupervised")
    parser.add_argument("--job_id", type=str, default="unknown", help="Job ID")
    parser.add_argument("--model_name", type=str, default="Llama2-MedTuned-13b", help="Name of base model to be used. Defaults to medbert. Must be saved in the path: paths.MODEL_PATH/model_name")
    parser.add_argument("--peft_model_name", type=str, default=None, help="Name of the PEFT model to be used. Defaults to None. Must be saved in the path: paths.MODEL_PATH/peft_model_name")
    parser.add_argument("--new_model_name", type=str, default=None, help="Directory to save the model. Defaults to model_name_finetuned")

    args = parser.parse_args()

    # Print or log the parsed arguments
    print("Parsed Arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    return args

def main():

    args = parse_args()
    JOB_ID = args.job_id
    MODEL_NAME = args.model_name
    PEFT_MODEL_NAME = args.peft_model_name
    NEW_MODEL_NAME = args.new_model_name

    base_model, tokenizer = load_model_and_tokenizer(
        model_name = MODEL_NAME,
        quantization = "float16",
        task_type = "clm",
    )

    model = PeftModel.from_pretrained(base_model, paths.MODEL_PATH/PEFT_MODEL_NAME)

    model = model.merge_and_unload()

    model.save_pretrained(paths.MODEL_PATH/NEW_MODEL_NAME)

    print(f"Model saved at {paths.MODEL_PATH/NEW_MODEL_NAME}")

    tokenizer.save_pretrained(paths.MODEL_PATH/NEW_MODEL_NAME)

    print(f"Tokenizer saved at {paths.MODEL_PATH/NEW_MODEL_NAME}")

    return

if __name__ == "__main__":
    main()
