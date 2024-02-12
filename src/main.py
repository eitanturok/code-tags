import argparse
import json
import os
from datetime import datetime

from datasets import load_dataset
from inference import inference, setup_inference
from model_output import clean_model_output
from prompt import get_prompt
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

            generated_function = clean_model_output(
                generation, prompt, example["declaration"], tokenizer, args.model_name
            )

            pass_tests = eval_code()

    get_score()


if __name__ == "__main__":
    args = get_args()
    output_dir = make_output_dir(args)
    fingerprint(args)
    set_seed(args.seed)

    small = True
    main(args, output_dir, small)
