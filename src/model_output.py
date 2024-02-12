def get_stop_words(tokenizer):
    stop_words = tokenizer.additional_special_tokens
    stop_words += ["<commit_before>", "<commit_msg>", "<commit_after>"]

    python_words = ["class", "def", "#", "@", "print", "if", "assert"]
    prefixes = ["\n"]
    separators = ["", " "]

    for python_word in python_words:
        for prefix in prefixes:
            for separator in separators:
                stop_words += [prefix + separator + python_word]
    return stop_words


def round_to_multiple(number, multiple):
    "A tab is usually 4 spaces."
    return multiple * round(number / multiple)


def clean_model_output(generation, prompt, declaration, tokenizer, model_name):

    # Initial values
    stop_words = get_stop_words(tokenizer)
    gen = generation

    # Get suffix length
    # Some models automatically append other tokens that are not in the prompt, so we remove those
    if model_name in ["llama", "codellama"]:
        suffix_length = len("<s> ")
    elif model_name in ["mpt", "starcoder"]:
        suffix_length = 0
    else:
        suffix_length = 0

    # Remove the initial prompt from the generation
    # This includes removing the function declaration (we'll put it back later)
    idx = len(prompt) + suffix_length
    gen = gen[idx:].rstrip()

    # Remove all text that occurs after the first stop_word
    for w in stop_words:
        if w in gen:
            gen = gen[: gen.find(w)]

    # 1 tab=4 whitespaces. Round number of leading whitespaces to closest multiple of 4.
    lines = []
    for i, line in enumerate(gen.split("\n")):
        n_whitespaces = len(line) - len(
            line.lstrip(" ")
        )  # only remove whitespace, not tabs
        new_whitespaces = " " * round_to_multiple(n_whitespaces, 4)
        line = new_whitespaces + line[n_whitespaces:]
        lines.append(line)
    gen = "\n".join(lines)

    ### Find the first occasion where a chain of { } is closed
    # Equivalently, find where the indentation of a python function ends
    # Adapted from https://github.com/bigcode-project/bigcode-evaluation-harness/blob/00967d12093ef614de7bdad0772aed8e4118f1fd/bigcode_eval/tasks/humanevalpack.py#L284
    for i, line in enumerate(gen.split("\n")):
        if len(line.strip()) > 0 and line[0] != " " and line[0] != "\t":
            gen = "\n".join(gen.split("\n")[:i])
            break

    # Define and save the generated function
    generated_function = declaration.rstrip() + "\n" + gen
    return generated_function
