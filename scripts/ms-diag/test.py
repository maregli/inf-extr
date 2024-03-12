import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Zero Shot Classification with Llama2-Chat")
    parser.add_argument("--job_id", type=str, default="unknown", help="Job ID")
    parser.add_argument("--model_name", type=str, default="medbert-512", help="Name of base model to be used. Defaults to medbert. Must be saved in the path: paths.MODEL_PATH/model_name")
    parser.add_argument("--quantization", type=str, default=None, help="Quantization. Must be one of 4bit, bfloat16, float16 or None. Defaults to None")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch Size. Defaults to 4")
    parser.add_argument("--gen_config", type=str, default=None, help="Generation Config. JSON-formatted configuration. Defaults to None in which case default config is used.")
    parser.add_argument("--prompt_strategies", type=str, nargs="+", default="all", help="Prompt Strategies. Must be one or more of zero_shot_vanilla, zero_shot_instruction, few_shot_vanilla, few_shot_instruction, two_steps, all. Defaults to all.")
    parser.add_argument("--data", type=str, default="line", help="Data. Must be one of line, all or all_first_line_last. Whether dataset consisting of single lines should be used or all text per rid.")
    parser.add_argument("--split", type=str, default="test", help="Split. Must be one of train, val, test or all. Defaults to test.")
    parser.add_argument("--attn_implementation", type=str, default=None, help="To implement Flash Attention 2 provide flash_attention_2. Defaults to None.")
    parser.add_argument("--information_retrieval", action="store_true", help="Whether to perform information retrieval. Defaults to False. If True, the model will be loaded from the information retrieval path.")

    args = parser.parse_args()

    # Print or log the parsed arguments
    print("Parsed Arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    return args

def main():
    args = parse_args()
    print("Running test.py with the following arguments:")
    print(args)

    print("Type of prompt_strategies:", type(args.prompt_strategies))
    print("Length of prompt_strategies:", len(args.prompt_strategies))
    if "all" in args.prompt_strategies:
        print("Prompt strategies contain all")

if __name__ == "__main__":
    main()