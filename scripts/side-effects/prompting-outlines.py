import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import os

import torch

from src import paths
from src.utils import (load_model_and_tokenizer, 
                       check_gpu_memory,
                       get_format_fun,
                        information_retrieval, 
                        get_sampler,
                        get_outlines_generator,
                        get_pydantic_schema,
                        format_prompt,
                        outlines_prompting,
)

from pydantic import BaseModel

import argparse

from datasets import Dataset

import json

import random

def parse_args():
    parser = argparse.ArgumentParser(description="Side effects extraction with Llama2")
    parser.add_argument("--job_id", type=str, default="unknown", help="Job ID")
    parser.add_argument("--model_name", type=str, default="medbert-512", help="Name of base model to be used. Defaults to medbert. Must be saved in the path: paths.MODEL_PATH/model_name")
    parser.add_argument("--quantization", type=str, default=None, help="Quantization. Must be one of 4bit, bfloat16, float16 or None. Defaults to None")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch Size. Defaults to 4")
    parser.add_argument("--prompt_strategies", type=str, nargs="+", default="all", help="Prompt Strategies. Must be one or more of zero_shot_vanilla, zero_shot_instruction, few_shot_vanilla, few_shot_instruction, all. Defaults to all.")
    parser.add_argument("--sampler", type=str, default = "greedy", help="Outlines Sampler to be used for generation. Must be one of greedy, multinomial or beam.")
    parser.add_argument("--attn_implementation", type=str, default=None, help="To implement Flash Attention 2 provide flash_attention_2. Defaults to None.")
    parser.add_argument("--information_retrieval", action="store_true", help="Whether to perform information retrieval. Defaults to False. If True, the model will be loaded from the information retrieval path.")
    parser.add_argument("--max_tokens", type=int, default = 1000, help="Maximum tokens to be generated per example. Defaults to 1000.")
    parser.add_argument("--num_examples", type=int, default = 10, help="Number of examples to be provided in few shot. Defaults to 10")

    args = parser.parse_args()

    # Print or log the parsed arguments
    print("Parsed Arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    return args

def main()->None:
    
    args = parse_args()
    JOB_ID = args.job_id
    MODEL_NAME = args.model_name
    QUANTIZATION = args.quantization
    BATCH_SIZE = args.batch_size
    PROMPT_STRATEGIES = args.prompt_strategies
    SAMPLER_NAME = args.sampler
    ATTN_IMPLEMENTATION = args.attn_implementation
    INFORMATION_RETRIEVAL = args.information_retrieval
    MAX_TOKENS = args.max_tokens
    NUM_EXAMPLES = args.num_examples

    # Check GPU Memory
    check_gpu_memory()

    # Load Model and Tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name = MODEL_NAME,
                                                task_type = "outlines",
                                                quantization = QUANTIZATION,
                                                attn_implementation = ATTN_IMPLEMENTATION,
                                                )

    check_gpu_memory()

    print("Loaded Model and Tokenizer")

    # Get sampler
    sampler = get_sampler(SAMPLER_NAME)

    # Get outlines schema
    schema = get_pydantic_schema("side_effects")

    # Get Outlines Generator
    generator = get_outlines_generator(model, sampler, task = "json", schema = schema)

    print("Got Outlines generator")

    # Load Data
    df = Dataset.load_from_disk(paths.DATA_PATH_PREPROCESSED/"side-effects/kisim_diagnoses_combined_se_sample")

    print("Loaded Data")

    if INFORMATION_RETRIEVAL:
        print("Performing Information Retrieval")
        # Load Retrieval Model
        retrieval_model, retrieval_tokenizer = load_model_and_tokenizer(model_name="line-label_medbert-512_class_pipeline", 
                                                quantization=None, 
                                                num_labels=8,
                                                task_type="class")
        
        # Retrieve relevant information for training and validation
        df = df.add_column("original_text", df["text"])
        text = information_retrieval(retrieval_model=retrieval_model,
                                            retrieval_tokenizer=retrieval_tokenizer,
                                            text=df["text"],
                                            label = "medms")
        
        df = df.remove_columns("text").add_column("text", text)

        df_no_med = df.filter(lambda example: len(example["text"]) == 0)
        df = df.filter(lambda example: len(example["text"]) > 0)

        del retrieval_model
        del retrieval_tokenizer



    # Load Task instructions, system prompt and examples
    with open(paths.DATA_PATH_PREPROCESSED/"side-effects/task_instruction.txt", "r") as f:
        task_instruction = f.read()

    with open(paths.DATA_PATH_PREPROCESSED/"side-effects/system_prompt.txt", "r") as f:
        system_prompt = f.read()

    with open(paths.DATA_PATH_PREPROCESSED/"side-effects/examples.json", "r") as f:
        examples = json.load(f)

    if NUM_EXAMPLES:
        examples = random.sample(examples, NUM_EXAMPLES)   

    # Prompt strategies
    if "all" in PROMPT_STRATEGIES:
        PROMPT_STRATEGIES = ["zero_shot_vanilla", "zero_shot_instruction", "few_shot_vanilla", "few_shot_instruction"]

    # Prompting
    for prompting_strategy in PROMPT_STRATEGIES:
        print(f"Prompting Strategy: {prompting_strategy}")

        format_fun = get_format_fun(prompting_strategy=prompting_strategy)

        input = format_prompt(text = df["text"], format_fun=format_fun, task_instruction = task_instruction, system_prompt = system_prompt, examples = examples)

        model_answers = outlines_prompting(text = input, generator=generator, batch_size=BATCH_SIZE, max_tokens=MAX_TOKENS)

        model_answers = [model_answer.model_dump_json() if isinstance(model_answer, BaseModel) else model_answer for model_answer in model_answers]

        results = {"model_answers": model_answers}
        results["rid"] = df["rid"]
        results["text"] = df["text"]

        # Add Information Retrieval Results for no_ms (no hidden states for this)
        if INFORMATION_RETRIEVAL:
            results["model_answers"].extend(["no information found"]*len(df_no_med))
            results["rid"].extend(df_no_med["rid"])
            results["text"].extend(df_no_med["text"])
            results["original_text"] = df["original_text"] + df_no_med["original_text"]


        filename = f"side-effects_outlines_{MODEL_NAME}_{QUANTIZATION}_{prompting_strategy}"

        if "few_shot" in prompting_strategy:
            filename += f"_{NUM_EXAMPLES}_examples"

        if INFORMATION_RETRIEVAL:
            filename += "_rag"

        os.makedirs(paths.RESULTS_PATH/"side-effects", exist_ok=True)
        torch.save(results, paths.RESULTS_PATH/f"side-effects/{filename}.pt")
        print("Saved Results")

    return

if __name__ == "__main__":
    main()


