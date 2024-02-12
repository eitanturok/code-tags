import argparse
import json
import os
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def get_args():

    # Setup
    time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--model_path", type=str, default="bigcode/starcoder")
    parser.add_argument("--model_name", type=str, default="starcoder")

    # General
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=f"./results/{time}")

    # Prompting
    parser.add_argument("--prompt-style", type=str, default="starcoder")
    parser.add_argument("--with_tests", action="store_true")
    parser.add_argument("--with_docs", action="store_true")
    parser.add_argument("--with_tags", action="store_true")

    # Generation
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_length", type=int, default=1800)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--max_time", type=float, default=120)

    # Eval
    parser.add_argument("--pass_at_k", type=int, default=1)

    return parser.parse_args()


def fingerprint(args):
    args_in_dict = vars(args)
    path = f"{args.output_dir}/args.jsonl"
    with open(path, "w") as f:
        json.dump(args_in_dict, f, indent=2)


def get_eval_dataset():
    # Load dataset from hugging face
    ds = load_dataset(
        "bigcode/humanevalpack", "python", split="test", trust_remote_code=True
    )
    examples = list(ds.to_pandas().T.to_dict().values())
    return examples


def setup_inference(model_path):

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        token="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",
    )
    # to be able to tokenize text in batches
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", token="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"
    )

    model.eval()
    return model, tokenizer


def get_prompt(example, with_docs, with_tests, with_tags):

    # Make prompt_base
    prompt_base = example["declaration"]
    if with_docs:
        docstring = example["prompt"][
            len(example["declaration"]) :
        ]  # example['prompt'] includes the docstring nicely formatted + declaration, so just remove declaration
        prompt_base = example["declaration"] + docstring

    # Make buggy function
    buggy_function = prompt_base + example["buggy_solution"]

    # Make user_message
    instruction = f'Fix bugs in {example["entry_point"]}.'
    test_str = f"\n{example['test']}" if with_tests else ""
    user_message = f"{buggy_function}{test_str}\n{instruction}"
    if with_tags:
        user_message = (
            f"<commit_before>{buggy_function}{test_str}<commit_msg>{instruction}"
        )

    # Get prompt
    prompt = format_for_model(model_name, user_message, prompt_base, with_tags)

    # Make correct function
    correct_function = prompt_base + example["canonical_solution"]

    # Update example
    example |= {
        "prompt_base": prompt_base,
        "buggy_function": buggy_function,
        "correct_function": correct_function,
        "instruction": instruction,
        "prompt": prompt,  # we are overwriting prompt, but we can ignore the old prompt value
        "model_name": model_name,
        "with_tags": with_tags,
        "with_docstring": with_docstring,
        "with_tests": with_tests,
    }

    return example


def main(args):

    model, tokenizer = setup_inference(args.model_path)
    examples = get_eval_dataset()

    for example in examples:

        prompt = get_prompt(example, args.with_docs, args.with_tests, args.with_tags)
        generations = inference()

        for generation in generations:

            generated_function = process_model_output()

            pass_tests = eval_code()

    get_score()


if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    fingerprint(args)
    set_seed(args.seed)

    main(args)
