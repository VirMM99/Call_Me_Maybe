import json
import sys
from pydantic import ValidationError
from basemodels import FunctionDefinitionCheck, PromptItem


class ParsingFileError(Exception):
    """Custom exception for wrong format in JSON file"""
    pass


def load_fn_definitions(path: str) -> list[FunctionDefinitionCheck]:
    """Load the function_definitions.json file
    Args:
        path (str): Path to the function definitions JSON file
    Returns:
        list[FunctionDefinitionCheck]: List of valid function definitions
    Raises:
        ParsingFileError: If the file does not exist, is invalid JSON,
            or root element is not a list
    """
    function_list: list[FunctionDefinitionCheck] = []
    try:
        with open(path, 'r', encoding="utf-8") as fn_def_file:
            data = json.load(fn_def_file)
    except FileNotFoundError:
        raise ParsingFileError(f"'{path}' does not exist") from None
    except json.JSONDecodeError:
        raise ParsingFileError(f"'{path}' contains invalid JSON") from None
    if not isinstance(data, list):
        raise ParsingFileError(
            f"Error: the JSON files {path} must contain a list at its root."
            )
    for item in data:
        try:
            definition: FunctionDefinitionCheck = FunctionDefinitionCheck(
                                                    **item)
            function_list.append(definition)
        except ValidationError:
            print(
                f"The funtion definition:\n{item}\n"
                "does not match the expected format and will be ignored",
                file=sys.stderr
                )
    return function_list


def load_prompts(path: str) -> list[PromptItem]:
    """Load and validates the prompts from the function_calling_test.json
    Args:
        path (str): Path to the prompts JSON file
    Returns:
        list[PromptItem]: List of valid prompt items
    Raises:
        ParsingFileError: If the file does not exist, is invalid JSON,
            or root element is not a list
        """
    prompt_list: list[PromptItem] = []
    try:
        with open(path, 'r', encoding="utf-8") as prompt_file:
            data = json.load(prompt_file)
    except FileNotFoundError:
        raise ParsingFileError(f"'{path}' does not exist") from None
    except json.JSONDecodeError:
        raise ParsingFileError(f"'{path}' contains invalid JSON") from None
    if not isinstance(data, list):
        raise ParsingFileError(
            f"Error: the JSON files {path} must contain a"
            " list at its root."
            )
    for item in data:
        try:
            prompt: PromptItem = PromptItem(**item)
            prompt_list.append(prompt)
        except ValidationError:
            print(
                f"Prompt\n{item}\n does not match the correct format"
                " and will be ignored",
                file=sys.stderr
                )
    return prompt_list


# def check_file_loaders():
#     print("Function definition:")
#     print(load_fn_definitions("data/input/functions_definition.json"))
#     print("Prompts")
#     print(load_prompts("data/input/function_calling_tests.json"))


# if __name__ == "__main__":
#     check_file_loaders()
