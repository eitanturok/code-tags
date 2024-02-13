import argparse
import json
import os
from datetime import datetime

import pandas as pd
from datasets import load_dataset
from icecream import ic
from inference import inference, setup_inference
from model_output import clean_model_output
from prompt import get_prompt
from run_code import does_code_pass_tests
from score import scorer
from tqdm import tqdm
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
    parser.add_argument("--n_samples", type=int, default=20)
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


def save_dict(path, my_dict):
    list_of_dicts = []
    for id_, val in my_dict.items():
        ex_id = id_.split("_")[0]
        gen_id = "" if len(id_.split("_")) < 2 else id_.split("_")[1]

        list_of_dicts.append({
            "ex_id": ex_id,
            "gen_id": gen_id,
            "val": val
        })

    df = pd.DataFrame(list_of_dicts)
    df.to_json(path, orient="records", lines=True)


def fingerprint(args):
    args_dict = vars(args)
    path = f"{args.output_dir}/args.jsonl"
    save_dict(path, args_dict)


def get_eval_dataset():
    # Load dataset from hugging face
    ds = load_dataset(
        "bigcode/humanevalpack", "python", split="test", trust_remote_code=True
    )
    examples = list(ds.to_pandas().T.to_dict().values())
    examples = {"ex" + str(i).zfill(2): example for i, example in enumerate(examples)}

    # Add docstring to example
    # example['prompt'] includes the docstring nicely formatted + declaration, so just remove declaration
    new_examples = {}
    for ex_id, example in examples.items():
        example["docstring"] = example["prompt"][len(example["declaration"]) :]
        new_examples[ex_id] = example
    examples = new_examples
    return examples


def main(args, output_dir, small):

    if small:
        args.max_time = 10
        args.n_samples = 2

    # Getting examples
    print("Making examples...", end="\t")
    examples = get_eval_dataset()
    if small:
        examples = {list(examples.keys())[0]: list(examples.values())[0]}
    save_dict(output_dir + "/01_examples.jsonl", examples)
    print("Done.")

    # Get prompts
    print("Making prompts...", end="\t")
    all_prompts = {}
    for ex_id, example in examples.items():
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
        all_prompts[ex_id] = prompt
    save_dict(output_dir + "/02_prompts.jsonl", all_prompts)
    print("Done.")

    # Model generates a response
    print("Generating model responses...", end="\t")
    all_generations = {}
    model, tokenizer = setup_inference(args.model_path)
    for ex_id, prompt in tqdm(all_prompts.items()):
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
        for i, generation in enumerate(generations):
            gen_id = "gen" + str(i).zfill(2)
            id_ = ex_id + "_" + gen_id
            all_generations[id_] = generation
    save_dict(output_dir + "/03_generations.jsonl", all_generations)
    print("Done.")

    # Clean model output
    print("Cleaning model generations...", end="\t")
    all_generated_functions = {}
    for id_, generation in all_generations.items():
        ex_id, gen_id = tuple(id_.split("_"))
        generated_function = clean_model_output(
            generation,
            prompt,
            examples[ex_id]["declaration"],
            tokenizer,
            args.model_name,
        )
        all_generated_functions[id_] = generated_function
    save_dict(output_dir + "/04_generated_functions.jsonl", all_generated_functions)
    print("Done.")

    # See if code passes the tests
    print("Running code on tests...", end="\t")
    all_pass_tests = {}
    for id_, generated_function in all_generated_functions.items():
        ex_id, gen_id = tuple(id_.split("_"))
        pass_tests = does_code_pass_tests(generated_function, examples[ex_id]["test"])
        all_pass_tests[id_] = pass_tests
    save_dict(output_dir + "/05_pass_tests.jsonl", all_pass_tests)
    print("Done.")

    # Score the results
    score = scorer(all_pass_tests, args.pass_at_k)
    score_dict = {f"pass@{args.pass_at_k}": score}
    save_dict(output_dir + "/06_score.jsonl", score_dict)
    print(f"{args.model_name} pass@{args.pass_at_k}={score}")

    print(f"Results in the directory: {output_dir}.")


if __name__ == "__main__":
    args = get_args()
    output_dir = make_output_dir(args)
    fingerprint(args)
    set_seed(args.seed)

    small = False
    main(args, output_dir, small)
