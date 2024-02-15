# Inspiration: https://github.com/bigcode-project/bigcode-evaluation-harness/blob/00967d12093ef614de7bdad0772aed8e4118f1fd/bigcode_eval/tasks/humanevalpack.py#L275

from typing import Dict, List

LANGUAGES: List[str] = ["python", "cpp", "js", "java", "go", "rust"]

# Taken from https://huggingface.co/datasets/nuprl/MultiPL-E/ & https://github.com/THUDM/CodeGeeX
LANGUAGE_TO_STOP_WORDS: Dict[str, List[str]] = {
    # https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L164
    "python": ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\nassert"],
    # https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L185
    "cpp": [],
    # https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L188
    "js": [],
    # https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L177
    "go": ["\n//", "\nfunc main(", "struct", "\nfunc"],
    # https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L169
    "java": [],
    "rust": [],
}


def prune_by_stop_word(generation: str, language: str) -> str:
    """Remove all text that appears after the first occurring stop word."""
    stop_words = LANGUAGE_TO_STOP_WORDS[language]
    for word in stop_words:
        if word in generation:
            generation = generation[: generation.find(word)]
    return generation

def prune_by_brackets(generation: str, language: str) -> str:
    """Find the first occasion where a chain of { } is closed and
    remove all text that appears after that. Equivalently in Python,
    find where the indentation is closed.
    """
    if language == "python":
        for i, line in enumerate(generation.split("\n")):
            if len(line.strip()) > 0 and line[0] != ' ' and line[0] != '\t':
                return "\n".join(generation.split("\n")[:i])
    elif language in ["java", "js", "go", "cpp", "rust"]:
        open_brackets = 2 if language == "java" else 1
        cut = False
        for i, c in enumerate(generation):
            if c == '{':
                open_brackets += 1
            elif c == '}':
                open_brackets -= 1
            if open_brackets == 0:
                generation = generation[:i+1]
                cut = True
                break
        if not cut:
            if language == "java":
                main_pos = generation.find("public static void main")
                if main_pos != -1:
                    generation = generation[:main_pos] + '}'
                if '}' in generation:
                    generation = generation[:generation.rfind('}')] + '}'
                if generation.count('{') - 1 == generation.count('}'):
                    generation += "\n}"
            elif '}' in generation:
                generation = generation[:generation.rfind('}')] + '}'
    return generation

def prompt_to_declaration(sample_prompt: str) -> str:
    idx = sample_prompt.find(":\n")
    function_declaration = sample_prompt[: idx + 1]
    function_declaration = function_declaration.rstrip('\n') + "\n"
    return function_declaration

def postprocess_code_generation(code_generation: str, language: str, sample_prompt) -> str:
    """Postprocess and clean up the LM generation of code.

    Args:
        code_generation (str): Code generation from LM.
        language (str): The coding language of the generation.
        max_seq_len (int): The prompt used to create the generation.

    Returns:
        final_code (list): The post-processed code generation.
    """

    code_generation = prune_by_stop_word(code_generation, language)
    code_generation = prune_by_brackets(code_generation, language)
    function_declaration = prompt_to_declaration(sample_prompt)
    final_code = function_declaration + code_generation
    return final_code
