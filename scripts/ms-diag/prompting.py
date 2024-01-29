
import torch
from torch.utils.data import DataLoader

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src import paths
from src.utils import (load_model_and_tokenizer, 
                       load_ms_data,  
                       check_gpu_memory, 
)

import argparse

from transformers import DataCollatorWithPadding, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, AutoModel

from datasets import concatenate_datasets, Dataset

from tqdm import tqdm

from typing import Callable

import json


def parse_args():
    parser = argparse.ArgumentParser(description="Zero Shot Classification with Llama2-Chat")
    parser.add_argument("--job_id", type=str, default="unknown", help="Job ID")
    parser.add_argument("--model_name", type=str, default="medbert-512", help="Name of base model to be used. Defaults to medbert. Must be saved in the path: paths.MODEL_PATH/model_name")
    parser.add_argument("--quantization", type=str, default=None, help="Quantization. Must be one of 4bit, bfloat16, float16 or None. Defaults to None")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch Size. Defaults to 4")
    parser.add_argument("--gen_config", type=str, default=None, help="Generation Config. JSON-formatted configuration. Defaults to None in which case default config is used.")
    parser.add_argument("--prompt_strategies", type=str, nargs="+", default="all", help="Prompt Strategies. Must be a list of zero_shot_vanilla, zero_shot_instruction, few_shot_vanilla, few_shot_instruction, two_steps. Defaults to zero_shot_vanilla.")
    parser.add_argument("--data", type=str, default="line", help="Data. Must be one of line, all or all_first_line_last. Whether dataset consisting of single lines should be used or all text per rid.")
    parser.add_argument("--split", type=str, default="test", help="Split. Must be one of train, val, test or all. Defaults to test.")
    parser.add_argument("--attn_implementation", type=str, default=None, help="To implement Flash Attention 2 provide flash_attention_2. Defaults to None.")

    args = parser.parse_args()

    # Print or log the parsed arguments
    print("Parsed Arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    return args


def get_hidden_state(text: list[str], model:AutoModel, tokenizer:AutoTokenizer, device:torch.device=torch.device("cpu"), batch_size:int = 16)->list[torch.Tensor]:
    """Get hidden state of last layer of model for each prediction in results. Per default the last 
    
    Args:
        text (list[str]): a list of input strings to be encoded.
        model (AutoModel): model
        tokenizer (AutoTokenizer): tokenizer
        device (torch.device): device. Defaults to torch.device("cpu").
        batch_size (int): batch size. Defaults to 16.
        
    Returns:
        results (dict): results of prompting return with keys report, prediction, last_hidden_states, input_lengths, whole_prompt, encodings
            
            
    """

    dataset = Dataset.from_dict(tokenizer(text, add_special_tokens = False))
    collate_fn = DataCollatorWithPadding(tokenizer = tokenizer, padding = "longest")

    dataloader = DataLoader(dataset = dataset, batch_size = batch_size, collate_fn = collate_fn)
    
    encodings = []

    for batch in tqdm(dataloader):
        batch.to(device)

        with torch.no_grad():
            outputs = model(**batch, output_hidden_states = True)

        last_hidden_state = outputs["hidden_states"][-1].to("cpu")

        # For decoder architectures the last token of the sequence contains information about the whole sequence
        last_hidden_state = last_hidden_state[:, -1, :]
        encodings.append(last_hidden_state)

        del last_hidden_state
        del outputs
        del batch
        torch.cuda.empty_cache()
        
    return torch.cat(encodings, dim=0)

def single_round_inference(reports:list[str], 
                           model:AutoModelForCausalLM, 
                           tokenizer:AutoTokenizer, 
                           format_fun:Callable[[str],str], 
                           generation_config:GenerationConfig = None,
                           prefix:str = "[/INST]",
                           device:torch.device = torch.device("cpu"),
                           batch_size:int=1,
                           output_hidden_states:bool = True)->dict:
    
    """ Single round inference for the MS extraction task
    
    Args:
        reports (list[str]): list of medical reports
        model (AutoModelForCausalLM): model
        tokenizer (AutoTokenizer): tokenizer
        format_fun (Callable[[str],str]): function to convert input text to desired prompt format
        generation_config (GenerationConfig): generation config. Defaults to None. If None, default config is used.
        prefix (str): prefix that separates input from output. Defaults to "[/INST]".
        device (torch.device): device. Defaults to torch.device("cpu").
        batch_size (int): batch size. Defaults to 1.
        output_hidden_states (bool); whether hidden states should be calculated for model answers. Defaults to True
        
    Returns:
        dict: dictionary with keys report, prediction, last_hidden_states, input_lengths, whole_prompt
            
    """
    print("Starting Inference")
    tokens = [tokenizer(format_fun(t), add_special_tokens = False, truncation = True) for t in reports]
    
    collate_fn = DataCollatorWithPadding(tokenizer, padding=True)

    dataloader = DataLoader(dataset=tokens, collate_fn=collate_fn, batch_size=batch_size, shuffle = False) 

    model.eval()

    results = []
    whole_prompt = []
    last_hidden_states = []
    input_lengths = [len(t["input_ids"]) for t in tokens]

    for idx, batch in enumerate(tqdm(dataloader)):
        batch.to(device)
        with torch.no_grad():
            outputs = model.generate(
                **batch,
                generation_config=generation_config,
            )

        # Check GPU memory every 5 batches
        if idx % 5 == 0:
            check_gpu_memory()


        return_tokens = outputs["sequences"].to("cpu")
        batch_result = tokenizer.batch_decode(return_tokens, skip_special_tokens=True)
        whole_prompt.extend(batch_result)
        batch_result = [result.split(prefix)[-1].lower().strip() for result in batch_result]

        results.extend(batch_result)
        del outputs
        torch.cuda.empty_cache()

    print("Finished Inference")

    if output_hidden_states:
        # This separate pass is not time but memory efficient as if the hidden states are calculated during inference,
        # hidden states are calculated for every token in the batch and the generated sequence.
        print("Starting Hidden State Calculation")
        last_hidden_states = get_hidden_state(results, model, tokenizer, device, batch_size)
        print("Finished Hidden State Calculation")

    else:
        last_hidden_states = None

        
    return {"report": reports, 
            "prediction": results, 
            "last_hidden_states": last_hidden_states, 
            "input_lengths":input_lengths,
            "whole_prompt": whole_prompt}

def multi_round_inference(reports:list[str], 
                          model:AutoModelForCausalLM, 
                          tokenizer:AutoTokenizer, 
                          format_fun1:Callable[[str],str],
                          format_fun2:Callable[[str],str],
                          generation_config:GenerationConfig = None,
                          prefix:str = "[/INST]",
                          device:torch.device = torch.device("cpu"),
                          batch_size:int = 1)->dict:
    
    """Multi Round inference for the MS extraction task
    
    Args:
        reports (list[str]): list of medical reports
        model (AutoModelForCausalLM): model
        tokenizer (AutoTokenizer): tokenizer
        format_fun1 (Callable[str,str]): function to convert input text to desired prompt format
        format_fun2 (Callable[str,str]): function to convert chat history to desired prompt format
        generation_config (GenerationConfig): generation config. Defaults to None. If None, default config is used.
        prefix (str): prefix that separates input from output. Defaults to "[/INST]".
        device (torch.device): device. Defaults to torch.device("cpu").
        batch_size (int): batch size. Defaults to 1.
        
    Returns:
        dict: dictionary with keys report, prediction, last_hidden_states, input_lengths, whole_prompt
            
    """
    
    max_new_tokens = generation_config.max_new_tokens

    # For first round don't calculate hidden states and use long max_new_tokens, and low batch size
    generation_config.max_new_tokens = 100
    output_round1 = single_round_inference(reports=reports,
                                           model=model, 
                                           tokenizer=tokenizer, 
                                           format_fun=format_fun1,
                                           generation_config=generation_config,
                                           device=device,
                                           batch_size=1,
                                           output_hidden_states=False)

    # For second round calculate hidden states if desired
    generation_config.max_new_tokens = max_new_tokens
    chat_history = output_round1["whole_prompt"]
    chat_history = [text.split("[/INST]")[-1] for text in chat_history]

    return single_round_inference(reports=chat_history,
                                  model=model, 
                                  tokenizer=tokenizer, 
                                  format_fun=format_fun2,
                                  generation_config=generation_config,
                                  prefix=prefix,
                                  device=device,
                                  batch_size=batch_size,
                                  output_hidden_states=True)

def zero_shot_base(report:str)->str:
    """Zero-shot base for the MS extraction task

    Args:
        report (str): medical report

    Returns:
        str: reformatted medical report with base

    """
    base_prompt = "<s>[INST]<<SYS>>{system_prompt}<</SYS>>\n\n{instruction}{input}[/INST]\nThe type of multiple sclerosis stated in the german medical report is: "
    system_prompt =  ("\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
                      "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
                       "Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make "
                        "any sense, or is not factually coherent, explain why instead of answering something not correct. If you don’t "
                        "know the answer to a question, please don’t share false information.\n"
                        )
    instruction = ("Your task is to extract the type of multiple Sclerosis (MS) stated in a German medical report. There are 3 types: "
                        "\"primär progrediente Multiple Sklerose (PPMS)\", \"sekundär progrediente Multiple Sklerose (SPMS)\" and \"schubförmig remittierende Multiple Sklerose (RRMS)\"."
                        "The type is provided in the text you just have to extract it. If you cannot match a type exactly answer with \"not enough info\"."
                        "Your answer should solely consist of either \"primär progrediente Multiple Sklerose (PPMS)\", \"sekundär progrediente Multiple Sklerose (SPMS)\" "
                        "\schubförmige remittierende Multiple Sklerose (RRMS)\", or \"not enough info\"."
                        "\nHere is the medical report:\n"
                    )
    input = base_prompt.format(system_prompt = system_prompt, instruction = instruction, input =  report)

    return input

def zero_shot_instruction(report:str)->str:
    """Zero-shot instruction for the MS extraction task
    
    Args:
        report (str): medical report
        
        Returns:
            str: reformatted medical report with instruction
            
            """
    instruction_base_prompt = "<s>[INST]\n### Instruction:\n{instruction}\n\n### Input:\n{input}[/INST]\n\n### Output:\n"
    task_instruction = ("Your task is to extract the type of multiple Sclerosis (MS) stated in a German medical report. There are 3 types: "
                        "\"primär progrediente Multiple Sklerose (PPMS)\", \"sekundär progrediente Multiple Sklerose (SPMS)\" and \"schubförmig remittierende Multiple Sklerose (RRMS)\"."
                        "The type is provided in the text you just have to extract it. If you cannot match a type exactly answer with \"not enough info\"."
                        "Your answer should solely consist of either \"primär progrediente Multiple Sklerose (PPMS)\", \"sekundär progrediente Multiple Sklerose (SPMS)\" "
                        "\schubförmige remittierende Multiple Sklerose (RRMS)\", or \"not enough info\"."
                        "\nHere is the medical report:\n"
                    )
    input = instruction_base_prompt.format(instruction = task_instruction, input =  report)

    return input

def few_shot_base(report:str)->str:
    """Few Shot base for the MS extraction task

    Args:
        report (str): medical report

    Returns:
        str: reformatted medical report with base

    """
    base_prompt = "<s>[INST]<<SYS>>{system_prompt}<</SYS>>\n\n{instruction}Report:\n{input}[/INST]\nDiagnosis:\n"

    rrms = 'Schubförmig-remittierende Multiple Sklerose, EM 01/2013, ED 10/2015\nINDENT EDSS 05/2020: 2.0 [...]'
    spms = '1. Sekundär progrediente schubförmige Multiple Sklerose [...]'
    ppms = '1. Primär progrediente Multiple Sklerose, EM 1992, ED 1996, aktuell EDSS 7.0 [...]'
    no_ms = '[...] INDENT MRI 07/2014: Progrediente supratentorielle MS-Plaques mit Befund-Progredienz im Bereich der Radiatio optica beidseits. [...]'

    examples = [ppms, spms, rrms, no_ms]

    labels = ["Schubförmig remittierende Multiple Sklerose (RRMS)",
              "Sekundär progrediente Multiple Sklerose (SPMS)",
              "Primär progrediente Multiple Sklerosis (PPMS)", 
              "not enough info"]
    
    system_prompt = (
    "\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
    "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
    "Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make "
    "any sense, or is not factually coherent, explain why instead of answering something not correct. If you don’t "
    "know the answer to a question, please don’t share false information.\n"
    )

    instruction = (
       "Your task is to extract the type of multiple Sclerosis (MS) stated in a German medical report. There are 3 types: "
        "\"primär progrediente Multiple Sklerose (PPMS)\", \"sekundär progrediente Multiple Sklerose (SPMS)\" and \"schubförmig remittierende Multiple Sklerose (RRMS)\"."
        "The type is provided in the text you just have to extract it. If you cannot match a type exactly answer with \"not enough info\"."
        "Your answer should solely consist of either \"primär progrediente Multiple Sklerose (PPMS)\", \"sekundär progrediente Multiple Sklerose (SPMS)\" "
        "\"schubförmige remittierende Multiple Sklerose (RRMS)\", or \"not enough info\"."
        "To help you with your task, here are a few excerpts from reports that indiciate what output you should produce:\n\n"
        )
    
    for example, label in zip(examples, labels):
        instruction += f"Report:\n{example}\nDiagnosis:\n{label}\n\n"
    
    input = base_prompt.format(system_prompt = system_prompt, instruction = instruction, input =  report)
    input + "Diagnosis:\n"

    return input

def few_shot_instruct(report:str)->str:
    """Few Shot base for the MS extraction task

    Args:
        report (str): medical report

    Returns:
        str: reformatted medical report with base

    """
    base_prompt = "<s>[INST]### Instruction:\n{instruction}### Input:\n{input}[/INST]\n### Output:\n"

    rrms = 'Schubförmig-remittierende Multiple Sklerose, EM 01/2013, ED 10/2015\nINDENT EDSS 05/2020: 2.0 [...]'
    spms = '1. Sekundär progrediente schubförmige Multiple Sklerose [...]'
    ppms = '1. Primär progrediente Multiple Sklerose, EM 1992, ED 1996, aktuell EDSS 7.0 [...]'
    no_ms = '[...] INDENT MRI 07/2014: Progrediente supratentorielle MS-Plaques mit Befund-Progredienz im Bereich der Radiatio optica beidseits. [...]'

    examples = [ppms, spms, rrms, no_ms]

    labels = ["Schubförmig remittierende Multiple Sklerose (RRMS)",
              "Sekundär progrediente Multiple Sklerose (SPMS)",
              "Primär progrediente Multiple Sklerosis (PPMS)", 
              "not enough info"]

    instruction = (
        "Your task is to extract the type of multiple Sclerosis (MS) stated in a German medical report. There are 3 types: "
        "\"primär progrediente Multiple Sklerose (PPMS)\", \"sekundär progrediente Multiple Sklerose (SPMS)\" and \"schubförmig remittierende Multiple Sklerose (RRMS)\"."
        "The type is provided in the text you just have to extract it. If you cannot match a type exactly answer with \"not enough info\"."
        "Your answer should solely consist of either \"primär progrediente Multiple Sklerose (PPMS)\", \"sekundär progrediente Multiple Sklerose (SPMS)\" "
        "\"schubförmige remittierende Multiple Sklerose (RRMS)\", or \"not enough info\"."
        "To help you with your task, here are a few excerpts from reports that indiciate what output you should produce:\n\n"
        )
    
    for example, label in zip(examples, labels):
        instruction += f"### Input:\n{example}\n### Output:\n{label}\n\n"
    
    input = base_prompt.format(instruction = instruction, input =  report)

    return input

def two_steps_one(report: str)->str:
    """Two Steps One for the MS extraction task. Encodes the report for first turn of the dialogue.

    Args:
        report (str): medical report

    Returns:
        str: reformatted medical report with base

    """
    base_prompt = "<s>[INST]<<SYS>>{system_prompt}<</SYS>>\n\n{instruction}{input}[/INST]"
    system_prompt =  ("\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
                      "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
                       "Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make "
                        "any sense, or is not factually coherent, explain why instead of answering something not correct. If you don’t "
                        "know the answer to a question, please don’t share false information.\n"
                        )
    instruction = ("Your task is to extract relevant information about the multiple sclerosis diagnosis from the provided German medical report. "
                   "Identify and summarize all sections discussing \"Multiple Sklerose\" paying attention to the exact type of multiple sclerosis. "
                   "There are three types:\n"
                   "primär progrediente Multiple Sklerose (PPMS)\n"
                   "sekundär progrediente Multiple Sklerose (SPMS)\n"
                   "schubförmige Multiple Sklerose (RRMS)\n"
                   "If the report lacks information about multiple sclerosis, respond with \"not enough info\". "
                   "\nHere is the Medical Report:\n "
                   )
                   
    input = base_prompt.format(system_prompt = system_prompt, instruction = instruction, input =  report)
    return input

def two_steps_two(chat_history: str)->str:
    """Two Steps Two for the MS extraction task. Encodes the chat history for second turn of the dialogue.

    Args:
        chat_history (str): chat history

    Returns:
        str: reformatted medical report with base

    """
    base_prompt = "<s>[INST]\n\n{instruction}\n{summary}[/INST]\nGiven the summary the most likely diagnosis is: "
    instruction = (
                   "Given a summary of a medical report describing a patient's condition related to multiple sclerosis, provide the most likely diagnosis. The possible diagnoses are:\n"
                   "- primär progrediente Multiple Sklerose (PPMS)\n"
                   "- sekundär progrediente Multiple Sklerose (SPMS)\n"
                   "- schubförmige Multiple Sklerose (RRMS)\n"
                   "- not enough info\n"
                   "Consider the information provided in the summary and select the diagnosis that best fits the patient's condition. If the summary does not contain sufficient information to make a diagnosis, choose \"not enough info.\" "
                   "Here is the summary:\n"
                   )
    input = base_prompt.format(instruction = instruction, summary = chat_history)

    return input


def get_format_fun(prompting_strategy:str)->Callable[[str],str]:
    """Get format function for prompting strategy

    Args:
        prompting_strategy (str): prompting strategy. Must be one of zero_shot_vanilla, zero_shot_instruction, few_shot_vanilla, few_shot_instruction, two_steps, or all. Defaults to zero_shot_vanilla.

    Returns:
        Callable[[str],str]: format function

    """
    if prompting_strategy == "zero_shot_vanilla":
        return zero_shot_base

    elif prompting_strategy == "zero_shot_instruction":
        return zero_shot_instruction

    elif prompting_strategy == "few_shot_vanilla":
        return few_shot_base

    elif prompting_strategy == "few_shot_instruction":
        return few_shot_instruct

    elif prompting_strategy == "two_steps":
        return two_steps_one

    else:
        raise ValueError(f"prompting_strategy must be one of zero_shot_vanilla, zero_shot_instruction, few_shot_vanilla, few_shot_instruction, two_steps, or all. Got {prompting_strategy}")
    
def get_prefix(prompting_strategy:str)->str:
    """Get prefix for prompting strategy

    Args:
        prompting_strategy (str): prompting strategy. Must be one of zero_shot_vanilla, zero_shot_instruction, few_shot_vanilla, few_shot_instruction, two_steps, or all. Defaults to zero_shot_vanilla.

    Returns:
        str: prefix

    """
    if prompting_strategy == "zero_shot_vanilla":
        return "[/INST]\nThe type of multiple sclerosis stated in the german medical report is: "

    elif prompting_strategy == "zero_shot_instruction":
        return "[/INST]\n\n### Output:\n"

    elif prompting_strategy == "few_shot_vanilla":
        return "[/INST]\nDiagnosis:\n"

    elif prompting_strategy == "few_shot_instruction":
        return "[/INST]\n### Output:\n"

    elif prompting_strategy == "two_steps":
        return "[/INST]\nGiven the summary the most likely diagnosis is: "

    else:
        return "[/INST]"
           
def prompting(prompting_strategy:str,
              reports:list[str],
              model:AutoModelForCausalLM,
              tokenizer:AutoTokenizer,
              generation_config:GenerationConfig,
              device:torch.device = torch.device("cpu"),
                batch_size:int = 1)->dict:
    """Prompting for the MS extraction task

    Args:
        prompting_strategy (str): prompting strategy. Must be one of zero_shot_vanilla, zero_shot_instruction, few_shot_vanilla, few_shot_instruction, two_steps, or all. Defaults to zero_shot_vanilla.
        reports (list[str]): list of medical reports
        model (AutoModelForCausalLM): model
        tokenizer (AutoTokenizer): tokenizer
        generation_config (GenerationConfig): generation config. Defaults to None. If None, default config is used.
        device (torch.device): device. Defaults to torch.device("cpu").
        batch_size (int): batch size. Defaults to 1.

    Returns:
        dict: dictionary with keys report, prediction, last_hidden_states, input_lengths, whole_prompt

    """

    format_fun = get_format_fun(prompting_strategy)
    prefix = get_prefix(prompting_strategy)

    if prompting_strategy == "two_steps":
        return multi_round_inference(reports=reports,
                                     model=model, 
                                     tokenizer=tokenizer, 
                                     format_fun1=format_fun,
                                     format_fun2=two_steps_two,
                                     generation_config=generation_config,
                                     prefix=prefix,
                                     device=device,
                                     batch_size=batch_size)
    else:
        return single_round_inference(reports=reports,
                                      model=model, 
                                      tokenizer=tokenizer, 
                                      format_fun=format_fun,
                                      generation_config=generation_config,
                                      prefix=prefix,
                                      device=device,
                                      batch_size=batch_size)
    


def main()->None:

    args = parse_args()
    JOB_ID = args.job_id
    MODEL_NAME = args.model_name
    QUANTIZATION = args.quantization
    GENERATION_CONFIG = args.gen_config
    PROMPT_STRATEGIES = args.prompt_strategies
    BATCH_SIZE = args.batch_size
    DATA = args.data
    SPLIT = args.split
    ATTN_IMPLEMENTATION = args.attn_implementation

    # Check GPU Memory
    check_gpu_memory()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model and Tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name = MODEL_NAME,
                                                task_type = "clm",
                                                quantization = QUANTIZATION,
                                                attn_implementation = ATTN_IMPLEMENTATION,
                                                )
    model.config.use_cache = True
    
    check_gpu_memory()

    print("Loaded Model and Tokenizer")

    # Load Data
    df = load_ms_data(data=DATA)

    if SPLIT == "all":
        df = concatenate_datasets([df["train"], df["val"], df["test"]])
    else:
        df = df[SPLIT]

    print("Loaded Data")

    # Encode Labels
    labels = ["Schubförmig remittierende Multiple Sklerose (RRMS)",
              "Sekundär progrediente Multiple Sklerose (SPMS)",
              "Primär progrediente Multiple Sklerosis (PPMS)", 
              "not enough info"]
    
    encoded_labels = get_hidden_state(labels, model, tokenizer, device)
    torch.save((labels, encoded_labels), paths.RESULTS_PATH/"ms-diag"/f"label_encodings_{MODEL_NAME}.pt")

    print("Saved Label Encodings")

    # Generation Config
    if GENERATION_CONFIG is not None:
        config = json.loads(GENERATION_CONFIG)
        GENERATION_CONFIG = GenerationConfig.from_dict(config)

    else:
        GENERATION_CONFIG = GenerationConfig(bos_token_id = tokenizer.bos_token_id,
                                     eos_token_id = tokenizer.eos_token_id,
                                     pad_token_id = tokenizer.pad_token_id,
                                     use_cache = True,
                                     max_new_tokens = 20,
                                     temperature=1,
                                     top_p=1,
                                     do_sample=False,
                                     output_hidden_states = False,
                                     return_dict_in_generate = True,
                                    )

    print("Loaded Generation Config")

    # Prompting
    for prompting_strategy in PROMPT_STRATEGIES:
        print(f"Prompting Strategy: {prompting_strategy}")
        results = prompting(prompting_strategy=prompting_strategy,
                            reports=df["text"], 
                            model=model, 
                            tokenizer=tokenizer, 
                            generation_config=GENERATION_CONFIG,
                            device=device,
                            batch_size=BATCH_SIZE)
        
        results["labels"] = df["labels"]
        results["rid"] = df["rid"]
        results["report"] = df["text"]

        filename = f"ms-diag_{MODEL_NAME}_{QUANTIZATION}_{DATA}_{SPLIT}_{prompting_strategy}"
        torch.save(results, paths.RESULTS_PATH/"ms-diag"/f"{filename}.pt")
        print("Saved Results")
    return

if __name__ == "__main__":
    main()


