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
