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
    parser.add_argument("--top_p", type=float, default=0.95)

    # Eval
    parser.add_argument("--pass_at_k", type=int, default=1)

    return parser.parse_args()


def make_output_dir(args):
    tag_str = "-tag" if args.with_tags else ""
    docstring_str = "-docstring" if args.with_docs else ""
    tests_str = "-test" if args.with_tests else ""
    output_dir = os.path.join(
        args.output_dir,
        f"{args.model_name}{tag_str}{docstring_str}{tests_str}",
    )
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


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

    # Add docstring to example
    # example['prompt'] includes the docstring nicely formatted + declaration, so just remove declaration
    new_examples = []
    for example in examples:
        example["docstring"] = example["prompt"][len(example["declaration"]) :]
        new_examples.append(example)
    examples = new_examples
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


def format_for_model(model_name, user_message, prompt_base, with_tags):
    tag_str = "<commit_after>" if with_tags else ""
    if model_name in ["mpt", "starcoder"]:
        prompt = f"{user_message}{tag_str}\n\n{prompt_base}"
    elif model_name in ["llama", "codellama"]:
        system_message = (
            "You are an expert programmer that helps to review Python code for bugs."
        )
        system_prompt = f"<<SYS>>\n{system_message}\n<</SYS>>"
        prompt = f"{system_prompt}[INST] {user_message}\n[/INST] {tag_str}{prompt_base}"
    else:
        prompt = f"{user_message}{tag_str}{prompt_base}"
    return prompt.strip()


def get_prompt(
    declaration,
    docstring,
    buggy_solution,
    entry_point,
    tests,
    model_name,
    with_docs,
    with_tests,
    with_tags,
):

    # Make prompt_base
    prompt_base = declaration
    if with_docs:
        prompt_base = declaration + docstring

    # Make buggy function
    buggy_function = prompt_base + buggy_solution

    # Make user_message
    instruction = f"Fix bugs in {entry_point}."
    test_str = f"\n{tests}" if with_tests else ""
    user_message = f"{buggy_function}{test_str}\n{instruction}"
    if with_tags:
        user_message = (
            f"<commit_before>{buggy_function}{test_str}<commit_msg>{instruction}"
        )

    # Get prompt
    prompt = format_for_model(model_name, user_message, prompt_base, with_tags)
    return prompt


def inference(
    prompt,
    model,
    tokenizer,
    n_samples,
    num_beams,
    temperature,
    max_length,
    top_p,
    max_time,
):

    if isinstance(prompt, str):
        prompt = [prompt]
    do_sample = True if top_p else False

    # Make n_samples of the prompt
    prompt *= n_samples

    # Tokenize
    encoded_inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(
        model.device
    )

    with torch.no_grad():
        encoded_outputs = model.generate(
            **encoded_inputs,
            max_length=max_length,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=num_beams,
            top_p=top_p,
            max_time=max_time,
        )
    texts = tokenizer.batch_decode(encoded_outputs)
    return texts


def main(args, output_dir, small):

    model, tokenizer = setup_inference(args.model_path)
    examples = get_eval_dataset()
    if small:
        examples = [examples[0]]
        args.max_time = 10

    for example in examples:

        prompt = get_prompt(
            example["declaration"],
            example["docstring"],
            example["buggy_solution"],
            example["entry_point"],
            example["test"],
            args.model_name,
            args.with_docs,
            args.with_tests,
            args.with_tags,
        )

        generations = inference(
            prompt,
            model,
            tokenizer,
            args.n_samples,
            args.num_beams,
            args.temperature,
            args.max_length,
            args.top_p,
            args.max_time,
        )

        if small:
            generations = [generations[0]]
        for generation in generations:

            generated_function = process_model_output()

            pass_tests = eval_code()

    get_score()


if __name__ == "__main__":
    args = get_args()
    output_dir = make_output_dir(args)
    fingerprint(args)
    set_seed(args.seed)

    small = True
    main(args, output_dir, small)
