import json
import sys
from pydantic import ValidationError
from .basemodels import FunctionDefinitionCheck, PromptItem


class ParsingFileError(Exception):
    """Exception raised when an input JSON file cannot be parsed"""
    pass


def load_fn_definitions(path: str) -> list[FunctionDefinitionCheck]:
    """Load and validate function definitions from a JSON file
    Args:
        path: Path to the function definitions JSON file
    Returns:
        A list of validated function definitions
    Raises:
        ParsingFileError: If the file does not exist,
            contains invalid JSON,
            or its root element is not a list
    """
    function_list: list[FunctionDefinitionCheck] = []
    try:
        with open(path, 'r', encoding="utf-8") as fn_def_file:
            data = json.load(fn_def_file)
    except FileNotFoundError:
        raise ParsingFileError(f"'{path}' does not exist") from None
    except json.JSONDecodeError:
        raise ParsingFileError(f"'{path}' contains invalid JSON") from None
    except OSError as error:  # This covers lecture problems with the file
        raise ParsingFileError(
                f"Could not read '{path}': {error}"
        ) from None

    if not isinstance(data, list):
        raise ParsingFileError(
            f"'{path}' must contain a list at its root."
            )

    for item in data:
        if not isinstance(item, dict):
            print(
                f"Function definition '{item}' is not a JSON object"
                " and will be ignored",
                file=sys.stderr,
            )
            continue

        try:
            definition = FunctionDefinitionCheck(**item)
            function_list.append(definition)
        except ValidationError as error:
            print(
                f"Invalid function definition:\n{item}\n"
                f"Reason: {error}",
                file=sys.stderr
                )
    return function_list


def load_prompts(path: str) -> list[PromptItem]:
    """Load and validates the prompts from a JSON file
    Args:
        path: Path to the prompts JSON file
    Returns:
        A list of validated prompts
    Raises:
        ParsingFileError: If the file does not exist,  cannot be read,
            contains Invalid JSON,
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
    except OSError as error:  # This covers lecture problems with the file
        raise ParsingFileError(
                f"Could not read '{path}': {error}"
        ) from None

    if not isinstance(data, list):
        raise ParsingFileError(
            f"'{path}' must contain a JSON"
            " list at its root."
            )

    for item in data:
        # This is so we can use item as dict (**item)
        if not isinstance(item, dict):
            print(
                f"Prompt '{item}' is not a JSON object"
                " and will be ignored",
                file=sys.stderr,
            )
            continue

        try:
            prompt = PromptItem(**item)
            prompt_list.append(prompt)
        # I use 'as error' to save what kind of error it is.
        # Very useful for debugging
        except ValidationError as error:
            print(
                f"Invalid prompt\n{item}\n"
                f"Reason: {error}",
                file=sys.stderr,
                )
    return prompt_list


# def check_file_loaders():
#     print("Function definition:")
#     print(load_fn_definitions("data/input/functions_definition.json"))
#     print("Prompts")
#     print(load_prompts("data/input/function_calling_tests.json"))


# if __name__ == "__main__":
#     check_file_loaders()
