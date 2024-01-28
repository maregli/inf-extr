
import torch

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

from datasets import concatenate_datasets

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
    parser.add_argument("--data", type=str, default="line", help="Data. Must be one of line, all or all_first_line_last. Whether dataset consisting of single lines should be used or all text per rid.")
    parser.add_argument("--split", type=str, default="test", help="Split. Must be one of train, val, test or all. Defaults to test.")
    parser.add_argument("--attn_implementation", type=str, default=None, help="To implement Flash Attention 2 provide flash_attention_2. Defaults to None.")

    args = parser.parse_args()

    # Print or log the parsed arguments
    print("Parsed Arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    return args

def label_encoding(labels:list[str], model:AutoModel, tokenizer:AutoTokenizer, device:torch.device=torch.device("cpu"))->list[torch.Tensor]:
    """Label encoding of labels
    
    Args:
        labels (list(str)): list of labels
        model (AutoModel): model
        tokenizer (AutoTokenizer): tokenizer
        device (torch.device): device. Defaults to torch.device("cpu").
        
    Returns:
        list(torch.Tensor): list of label encodings
            
            
    """
    encodings = {label:[] for label in labels}

    for label in labels:
        input = tokenizer(label, return_tensors = "pt", add_special_tokens = False)
        input.to(device)
        with torch.no_grad():
            outputs = model(**input, output_hidden_states = True)
        last_hidden_state = outputs["hidden_states"][-1]
        last_hidden_state = torch.mean(last_hidden_state, dim = 1)
        encodings[label] = last_hidden_state.to("cpu").squeeze()
        del last_hidden_state
        del outputs
        del input
        torch.cuda.empty_cache()

    return encodings

def single_round_inference(reports:list[str], 
                           model:AutoModelForCausalLM, 
                           tokenizer:AutoTokenizer, 
                           format_fun:Callable[[str],str], 
                           generation_config:GenerationConfig = None,
                           device:torch.device = torch.device("cpu"),
                           batch_size:int=1)->dict:
    
    """ Single round inference for the MS extraction task
    
    Args:
        reports (list[str]): list of medical reports
        model (AutoModelForCausalLM): model
        tokenizer (AutoTokenizer): tokenizer
        format_fun (Callable[[str],str]): function to convert input text to desired prompt format
        generation_config (GenerationConfig): generation config. Defaults to None. If None, default config is used.
        device (torch.device): device. Defaults to torch.device("cpu").
        
    Returns:
        dict: dictionary with keys report, prediction, last_hidden_states, input_lengths, whole_prompt
            
    """

    tokens = [tokenizer(format_fun(t), add_special_tokens = False, truncation = True) for t in reports]
    
    collate_fn = DataCollatorWithPadding(tokenizer, padding=True)

    dataloader = torch.utils.data.DataLoader(dataset=tokens, collate_fn=collate_fn, batch_size=batch_size, shuffle = False) 

    model.eval()

    results = []
    whole_prompt = []
    last_hidden_states = []
    input_lengths = [len(t["input_ids"]) for t in tokens]

    for idx, batch in tqdm(enumerate(dataloader), total = len(dataloader)):
        batch.to(device)
        with torch.no_grad():
            outputs = model.generate(
                **batch,
                generation_config=generation_config,
            )

        if idx % 5 == 0:
            check_gpu_memory()

        if generation_config.output_hidden_states:
            for idx in range(len(outputs.sequences)):
                # Find the index of eos_token_id in generated tokens if it exists
                eos_index = torch.where(outputs.sequences[idx] == tokenizer.eos_token_id)[0]
                # If eos_token_id does not exist in generated tokens, set to -1
                eos_index = eos_index[-1] if eos_index.numel() > 0 else -1
    
                # Extract the last hidden states for all the tokens in the output sequence
                # outputs["hidden_states"][:eos_index] is a tuple of tuples of hidden states (one for each layer) for all the generated tokens in the output sequence, it has length of generated sequence
                response_last_hidden_states_tuples = [hidden_state[-1][idx,:,:] for hidden_state in outputs["hidden_states"][:eos_index]]
                mean_last_hidden_states = torch.mean(torch.cat(response_last_hidden_states_tuples), dim=0)
                last_hidden_states.append(mean_last_hidden_states.to("cpu"))
        else:
            last_hidden_states.append([None] * len(outputs.sequences))


        return_tokens = outputs["sequences"].to("cpu")
        batch_result = tokenizer.batch_decode(return_tokens, skip_special_tokens=True)
        whole_prompt.extend(batch_result)
        batch_result = [result.split("[/INST]")[-1].lower().strip() for result in batch_result]

        results.extend(batch_result)
        del outputs
        torch.cuda.empty_cache()

        
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
                          device:torch.device = torch.device("cpu"),
                          batch_size:int = 1)->dict:
    
    """Multi Round inference for the MS extraction task
    
    Args:
        reports (list[str]): list of medical reports
        model (AutoModelForCausalLM): model
        tokenizer (AutoTokenizer): tokenizer
        format_fun1 (Callable[str,str]): function to convert input text to desired prompt format
        format_fun2 (Callable[str,str]): function to convert chat history to desired prompt format
        output_hidden_states (bool); whether hidden states should be calculated. Defaults to True
        max_new_tokens (int): The number of tokens to be generated.
        
    Returns:
        pd.DataFrame: results of inference
            
    """
    output_hidden_states = generation_config.output_hidden_states
    max_new_tokens = generation_config.max_new_tokens

    # For first round don't calculate hidden states and use long max_new_tokens
    generation_config.output_hidden_states = False
    generation_config.max_new_tokens = 100
    output_round1 = single_round_inference(reports=reports,
                                           model=model, 
                                           tokenizer=tokenizer, 
                                           format_fun=format_fun1,
                                           generation_config=generation_config,
                                           device=device,
                                           batch_size=1)

    # For second round calculate hidden states if desired
    generation_config.output_hidden_states = output_hidden_states
    generation_config.max_new_tokens = max_new_tokens
    chat_history = output_round1["whole_prompt"]

    return single_round_inference(reports=chat_history,
                                  model=model, 
                                  tokenizer=tokenizer, 
                                  format_fun=format_fun2,
                                  generation_config=generation_config,
                                  device=device,
                                  batch_size=batch_size)


def zero_shot_base(report:str)->str:
    """Zero-shot base for the MS extraction task

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
    instruction = ("Your task is to extract the type of multiple Sclerosis (MS) stated in a German medical report. There are 3 types: "
                    "primär progrediente Multiple Sklerose (PPMS), sekundär progrediente Multiple Sklerose (SPMS) and schubförmige Multiple Sklerose (RRMS)."
                    "The type is provided in the text you just have to extract it. If you cannot match a type exactly answer with \"not enough info\"."
                    "Your answer should solely consist of either \"primär progrediente Multiple Sklerose (PPMS)\", \"sekundär progrediente Multiple Sklerose (SPMS)\" "
                    "\schubförmige Multiple Sklerose (RRMS)\", or \"not enough info\"."
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
    instruction_base_prompt = "<s>[INST]\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Output:\n[/INST]"
    task_instruction = ("Your task is to extract the type of multiple Sclerosis (MS) stated in a German medical report. There are 3 types: "
                        "primär progrediente Multiple Sklerose (PPMS), sekundär progrediente Multiple Sklerose (SPMS) and schubförmige Multiple Sklerose (RRMS)."
                        "The type is provided in the text you just have to extract it. If you cannot match a type exactly answer with \"not enough info\"."
                        "Your answer should solely consist of either \"primär progrediente Multiple Sklerose (PPMS)\", \"sekundär progrediente Multiple Sklerose (SPMS)\" "
                        "\schubförmige Multiple Sklerose (RRMS)\", or \"not enough info\"."
                        "Here is the medical report: "
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
    base_prompt = "<s>[INST]<<SYS>>{system_prompt}<</SYS>>\n\n{instruction}Report:\n{input}\nDiagnosis:\n[/INST]"

    rrms = 'Schubförmig-remittierende Multiple Sklerose, EM 01/2013, ED 10/2015\nINDENT EDSS 05/2020: 2.0 [...]'
    spms = '1. Sekundär progrediente schubförmige Multiple Sklerose [...]'
    ppms = '1. Primär progrediente Multiple Sklerose, EM 1992, ED 1996, aktuell EDSS 7.0 [...]'
    no_ms = '[...] INDENT MRI 07/2014: Progrediente supratentorielle MS-Plaques mit Befund-Progredienz im Bereich der Radiatio optica beidseits. [...]'

    examples = [ppms, spms, rrms, no_ms]

    labels = ["primary progressive multiple sclerosis", 
              "secondary progressive multiple sclerosis",
              "relapsing remitting multiple sclerosis",
              "no multiple sclerosis"]
    
    system_prompt = (
    "\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
    "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
    "Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make "
    "any sense, or is not factually coherent, explain why instead of answering something not correct. If you don’t "
    "know the answer to a question, please don’t share false information.\n"
    )

    instruction = (
       "Your task is to extract the type of multiple Sclerosis (MS) stated in a German medical report. There are 3 types: "
        "primär progrediente Multiple Sklerose (PPMS), sekundär progrediente Multiple Sklerose (SPMS) and schubförmige Multiple Sklerose (RRMS)."
        "The type is provided in the text you just have to extract it. If you cannot match a type exactly answer with \"not enough info\"."
        "Your answer should solely consist of either \"primär progrediente Multiple Sklerose (PPMS)\", \"sekundär progrediente Multiple Sklerose (SPMS)\" "
        "\schubförmige Multiple Sklerose (RRMS)\", or \"not enough info\"."
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
    base_prompt = "<s>[INST]### Instruction:\n{instruction}### Input:\n{input}\n### Output:\n[/INST]"

    rrms = 'Schubförmig-remittierende Multiple Sklerose, EM 01/2013, ED 10/2015\nINDENT EDSS 05/2020: 2.0 [...]'
    spms = '1. Sekundär progrediente schubförmige Multiple Sklerose [...]'
    ppms = '1. Primär progrediente Multiple Sklerose, EM 1992, ED 1996, aktuell EDSS 7.0 [...]'
    no_ms = '[...] INDENT MRI 07/2014: Progrediente supratentorielle MS-Plaques mit Befund-Progredienz im Bereich der Radiatio optica beidseits. [...]'

    examples = [ppms, spms, rrms, no_ms]

    labels = ["primary progressive multiple sclerosis", 
              "secondary progressive multiple sclerosis",
              "relapsing remitting multiple sclerosis",
              "not enough info"]

    instruction = (
        "Your task is to extract the type of multiple Sclerosis (MS) stated in a German medical report. There are 3 types: "
        "primär progrediente Multiple Sklerose (PPMS), sekundär progrediente Multiple Sklerose (SPMS) and schubförmige Multiple Sklerose (RRMS)."
        "The type is provided in the text you just have to extract it. If you cannot match a type exactly answer with \"not enough info\"."
        "Your answer should solely consist of either \"primär progrediente Multiple Sklerose (PPMS)\", \"sekundär progrediente Multiple Sklerose (SPMS)\" "
        "\schubförmige Multiple Sklerose (RRMS)\", or \"not enough info\"."
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
    instruction = ("Your task is to summarize all relevant information pertaining to the multiple sclerosis diagnosis "
                    "from the provided German medical report. The German word for multiple sclerosis is: \"Multiple Sklerose\", "
                    "watch for this keyword and extract all the text around it, especially words before and after. "
                    "If the report contains no information regarding multiple sclerosis, "
                    "please respond with \"not enough info.\" "
                    "\nHere is the medical report:\n\n"
                   )
    input = base_prompt.format(system_prompt = system_prompt, instruction = instruction, input =  report)
    return input

def two_steps_two(chat_history: str, eos_token:str='</s>')->str:
    """Two Steps Two for the MS extraction task. Encodes the chat history for second turn of the dialogue.

    Args:
        chat_history (str): chat history
        eos_token (str): eos token. Defaults to '</s>'.

    Returns:
        str: reformatted medical report with base

    """
    base_prompt = "<s>[INST]\n\n{instruction}[/INST]"
    instruction = ("Given your summary of the medical report, which of the following is the most likely label for this report: "
                  "\"primär progrediente Multiple Sklerose (PPMS)\", \"sekundär progrediente Multiple Sklerose (SPMS)\", "
                   "\"schubförmige Multiple Sklerose (RRMS)\", or \"not enough info\". Your answer should have only consist of one of the mentioned labels."
                   )
    if not chat_history.endswith(eos_token):
        chat_history += eos_token
    input = chat_history + base_prompt.format(instruction = instruction)

    return input
                    

def main()->None:

    args = parse_args()
    JOB_ID = args.job_id
    MODEL_NAME = args.model_name
    QUANTIZATION = args.quantization
    GENERATION_CONFIG = args.gen_config
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
    labels = ["primary progressive multiple sclerosis", 
              "secondary progressive multiple sclerosis",
              "relapsing remitting multiple sclerosis",
              "not enough info"]
    
    encoded_labels = label_encoding(labels, model, tokenizer)

    torch.save(encoded_labels, paths.RESULTS_PATH/"ms-diag"/f"label_encodings_{MODEL_NAME}.pt")

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


    file_name_base = f"ms-diag_{MODEL_NAME}_{QUANTIZATION}_{DATA}_{SPLIT}_"

    # Zero Shot Vanilla
    filename = file_name_base + "zero_shot_vanilla"

    results = single_round_inference(reports=df["text"], 
                                     model=model, 
                                     tokenizer=tokenizer, 
                                     format_fun=zero_shot_base,
                                     generation_config=GENERATION_CONFIG,
                                     device=device,
                                     batch_size=BATCH_SIZE)
    
    torch.save(results, paths.RESULTS_PATH/"ms-diag"/f"{filename}.pt")

    # Zero Shot Instruction
    filename = file_name_base + "zero_shot_instruction"

    results = single_round_inference(reports=df["text"], 
                                     model=model, 
                                     tokenizer=tokenizer, 
                                     format_fun=zero_shot_instruction,
                                     generation_config=GENERATION_CONFIG,
                                     device=device,
                                     batch_size=BATCH_SIZE)

    torch.save(results, paths.RESULTS_PATH/"ms-diag"/f"{filename}.pt")

    # Few Shot Vanilla
    filename = file_name_base + "few_shot_vanilla"

    results = single_round_inference(reports=df["text"],
                                     model=model, 
                                     tokenizer=tokenizer, 
                                     format_fun=few_shot_base,
                                     generation_config=GENERATION_CONFIG,
                                     device=device,
                                     batch_size=BATCH_SIZE)
    
    torch.save(results, paths.RESULTS_PATH/"ms-diag"/f"{filename}.pt")

    # Few Shot Instruction
    filename = file_name_base + "few_shot_instruction"

    results = single_round_inference(reports=df["text"],
                                     model=model, 
                                     tokenizer=tokenizer, 
                                     format_fun=few_shot_instruct,
                                     generation_config=GENERATION_CONFIG,
                                     device=device,
                                     batch_size=BATCH_SIZE)
    
    torch.save(results, paths.RESULTS_PATH/"ms-diag"/f"{filename}.pt")

    # Two Steps
    filename = file_name_base + "two_steps"

    results = multi_round_inference(reports=df["text"],
                                    model=model, 
                                    tokenizer=tokenizer, 
                                    format_fun1=two_steps_one,
                                    format_fun2=two_steps_two,
                                    generation_config=GENERATION_CONFIG,
                                    device=device,
                                    batch_size=BATCH_SIZE)
    
    torch.save(results, paths.RESULTS_PATH/"ms-diag"/f"{filename}.pt")

    print("Saved Results")

    return

if __name__ == "__main__":
    main()


