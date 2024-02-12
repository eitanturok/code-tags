# %%
import multiprocessing
import os
import threading
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# %% [markdown]
# # Utils


# %%
def pp(examples):
    "pretty print the example(s)"
    if isinstance(examples, dict):
        examples = [examples]

    # For some reason, in this display angle brackets, i.e. <commit_msg>, does not appear unless you have a space seperating the < and >.
    def fix_bracket(text):
        return (
            text.replace("<code_before>", "< code_before >")
            .replace("<commit_msg>", "< commit_msg >")
            .replace("<code_after>", "< code_after >")
        )

    examples = [
        {k: fix_bracket(v) if isinstance(v, str) else v for k, v in example.items()}
        for example in examples
    ]

    # Turn to dataframe
    df = pd.DataFrame(examples)

    # Select subset of columns
    cols = [
        "buggy_function",
        "correct_function",
        "generated_function",
        "generation",
        "pass_tests",
    ]
    my_cols = [col for col in cols if col in df.columns.to_list()]
    if my_cols:
        df = df[my_cols]

    df = df.T

    # Turn into the visible format
    output = df.style.set_properties(
        **{
            "text-align": "left",
            "white-space": "pre-wrap",
        }
    )
    return output


# %%
def record_it(example, out_dir):
    tag_str = "-tag" if example["with_tags"] else ""
    docstring_str = "-docstring" if example["with_docstring"] else ""
    tests_str = "-test" if example["with_tests"] else ""
    path = os.path.join(
        "results",
        f"{example['model_name']}{tag_str}{docstring_str}{tests_str}",
        f"{out_dir}.jsonl",
    )

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([example])
    df.to_json(path, lines=True, orient="records", mode="a")


# %% [markdown]
# # Process Model Input
#
# Create a dataset which keeps track of
#
# 1. model_input
# 2. model_output
# 3. tests
# 4. declaration
# 5. commit_msg
# 6. cannonical_solution
# 7. buggy_solution
# 8. generated_solution
# 9. docs
#
#
# solution -> the code for the solution but no function declaration and (optionally) no docstring
# function -> the code for the solution with the function declaration and (optionally) the docstring
#
# For each model keep track of
# 1. system prompt


# %%
def format_for_model(model_name, user_message, prompt_base, with_tags):
    tag_str = "<code_after>" if with_tags else ""
    if model_name in ["mpt", "starcoder"]:
        prompt = f"{user_message}{tag_str}{prompt_base}"
    elif model_name in ["llama", "codellama"]:
        system_message = (
            "You are an expert programmer that helps to review Python code for bugs."
        )
        system_prompt = f"<<SYS>>\n{system_message}\n<</SYS>>"
        prompt = f"{system_prompt}[INST] {user_message}\n[/INST] {tag_str}{prompt_base}"
    else:
        prompt = f"{user_message}{tag_str}{prompt_base}"
    return prompt.strip()


# %%
def process_model_input(
    example, model_name=None, with_tags=False, with_docstring=False, with_tests=False
):

    # Make prompt_base
    prompt_base = example["declaration"]
    if with_docstring:
        docstring = example["prompt"][
            len(example["declaration"]) :
        ]  # example['prompt'] includes the docstring nicely formatted + declaration, so just remove declaration
        prompt_base = example["declaration"] + docstring

    # Make buggy function
    buggy_function = prompt_base + example["buggy_solution"]

    # Make user_message
    instruction = f'Fix bugs in {example["entry_point"]}.'
    test_str = f"{example['test']}" if with_tests else ""
    user_message = f"{buggy_function}{test_str}\n{instruction}"
    if with_tags:
        user_message = (
            f"<code_before>{buggy_function}{test_str}<commit_msg>{instruction}"
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


# %% [markdown]
# # Model Inference


# %%
def initialize_model(model_path):
    print(f"Initializing {model_path}")
    trust_remote_code = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        token="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",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", token="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"
    )

    model.eval()
    # model.to(device)
    return model, tokenizer


# %%
def inference(prompt, model, tokenizer, num_beams=5):
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=600,
        eos_token_id=100257,
        do_sample=True,
        temperature=0.2,
        pad_token_id=tokenizer.eos_token_id,
        num_beams=num_beams,
        max_time=30,
    )
    text = tokenizer.batch_decode(outputs)[0]
    return text


# %% [markdown]
# # Clean Output


# %%
def get_stop_words(tokenizer):
    stop_words = tokenizer.additional_special_tokens
    stop_words += ["<commit_before>", "<commit_msg>", "<commit_after>"]
    stop_words += ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\nassert"]
    return stop_words


# %%
def round_to_multiple(number, multiple):
    "A tab is usually 4 spaces."
    return multiple * round(number / multiple)


# %%
def process_model_output(example, tokenizer):

    # Initial values
    stop_words = get_stop_words(tokenizer)
    gen = example["generation"]

    # Remove the initial prompt from the generation
    idx = len(example["prompt"])
    gen = gen[idx:].rstrip()

    # Remove all text that occurs after the first stop_word
    for w in stop_words:
        if w in gen:
            gen = gen[: gen.find(w)]

    lines = []
    for i, line in enumerate(gen.split("\n")):

        n_whitespaces = len(line) - len(line.lstrip())
        new_whitespaces = " " * round_to_multiple(n_whitespaces, 4)
        line = new_whitespaces + line[n_whitespaces:]
        lines.append(line)
    gen = "\n".join(lines)

    generated_function = example["declaration"].rstrip() + gen
    example["generated_function"] = generated_function
    return example


# %% [markdown]
# # Eval Code


# %%
def timeout_handler(process):
    if process.is_alive():
        print("Timeout reached! Cancelling the function.")
        process.terminate()


def run_function_with_timeout(target_function, args=(), timeout_seconds=5):
    # Initialize return_dict for us to store values in
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    return_dict["pass_tests"] = False

    # Create a wrapper function to handle storing the right value in return_dict
    def target_function_wrapper(target_function, return_dict, *args):
        try:
            target_function(*args)
            return_dict["pass_tests"] = True
        except Exception as e:
            return_dict["pass_tests"] = False

    # Define the process
    process = multiprocessing.Process(
        target=target_function_wrapper, args=(target_function, return_dict) + args
    )

    # Start the process
    process.start()

    # Set up a timer to terminate the process if it exceeds the timeout
    timer = threading.Timer(timeout_seconds, timeout_handler, args=(process,))
    timer.start()

    # Wait for the process to finish
    process.join()

    # Cancel the timer
    timer.cancel()
    return return_dict.values()[0]


# %%
def does_code_pass_tests(code) -> bool:
    return run_function_with_timeout(exec, args=(code,))


# %%
def eval_code(example):
    code = example["generated_function"] + example["test"]
    example["pass_tests"] = does_code_pass_tests(code)
    return example


# %% [markdown]
# # Main

# %%
models = {
    # "codellama": "codellama/CodeLlama-7b-hf",
    # "starcoder": "bigcode/starcoder",
    "mpt": "../my_hf_model",
    # "llama": "meta-llama/Llama-2-7b-hf",
}
with_tags = False

# %%
ds = load_dataset(
    "bigcode/humanevalpack", "python", split="test", trust_remote_code=True
)

# %%
for model_name, model_path in models.items():
    examples = list(ds.to_pandas().T.to_dict().values())
    model, tokenizer = initialize_model(model_path)

    # # Skip if already computed
    # path = os.path.join("results", model_name, "04_code_eval.jsonl")
    # if os.path.exists(path):
    #     continue

    for example in tqdm(examples):
        example = process_model_input(
            example, model_name=model_name, with_tags=with_tags
        )
        record_it(example, "01_processed_data")

        example["generation"] = inference(example["prompt"], model, tokenizer)
        record_it(example, "02_model_generation")

        example = process_model_output(example, tokenizer)
        record_it(example, "03_processed_generation")

        example = eval_code(example)
        record_it(example, "04_code_eval")
