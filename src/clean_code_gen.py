def round_to_multiple(number, multiple):
    "A tab is usually 4 spaces."
    return multiple * round(number / multiple)


def get_stop_words():
    stop_words = ["<commit_before>", "<commit_msg>", "<commit_after>"]

    python_words = ["class", "def", "#", "@", "print", "if", "assert"]
    prefixes = ["\n"]
    separators = ["", " "]

    for python_word in python_words:
        for prefix in prefixes:
            for separator in separators:
                stop_words += [prefix + separator + python_word]
    return stop_words


def clean_code_gen(code_gen):

    # Remove all text that occurs after the first stop_word
    stop_words = get_stop_words()
    for w in stop_words:
        if w in code_gen:
            code_gen = code_gen[: code_gen.find(w)]

    # 1 tab=4 whitespaces. Round number of leading whitespaces to closest multiple of 4.
    lines = []
    for i, line in enumerate(code_gen.split("\n")):
        n_whitespaces = len(line) - len(
            line.lstrip(" ")
        )  # only remove whitespace, not tabs
        new_whitespaces = " " * round_to_multiple(n_whitespaces, 4)
        line = new_whitespaces + line[n_whitespaces:]
        lines.append(line)
    code_gen = "\n".join(lines)

    ### Find the first occasion where a chain of {} is closed
    # Equivalently, find where the indentation of a python function ends
    # Adapted from https://github.com/bigcode-project/bigcode-evaluation-harness/blob/00967d12093ef614de7bdad0772aed8e4118f1fd/bigcode_eval/tasks/humanevalpack.py#L284
    for i, line in enumerate(code_gen.split("\n")):
        if len(line.strip()) > 0 and line[0] != " " and line[0] != "\t":
            code_gen = "\n".join(code_gen.split("\n")[:i])
            break

    return code_gen.strip()


def get_declaration(sample_prompt):
    idx = sample_prompt.find(":\n")
    declaration = sample_prompt[: idx + 1]
    return declaration
