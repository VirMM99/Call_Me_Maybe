from src.basemodels import FunctionCallOutput, FunctionDefinitionCheck
from src.validator import (
    FunctionCallValidationError,
    validate_function_call,
)


def main() -> None:
    """Test function call validation."""
    definition = FunctionDefinitionCheck(
        name="fn_add_numbers",
        description="Add two numbers together.",
        parameters={
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        returns={"type": "number"},
    )

    greet_definition = FunctionDefinitionCheck(
        name="fn_greet",
        description="Generate a greeting message for a person by name.",
        parameters={
            "name": {"type": "string"},
        },
        returns={"type": "string"},
    )

    valid_call = FunctionCallOutput(
        prompt="What is the sum of 2 and 3?",
        name="fn_add_numbers",
        parameters={
            "a": 2,
            "b": 3,
        },
    )
    missing_parameter_call = FunctionCallOutput(
        prompt="What is the sum of 2 and 3?",
        name="fn_add_numbers",
        parameters={
            "a": 2,
        },
    )

    extra_parameter_call = FunctionCallOutput(
        prompt="What is the sum of 2 and 3?",
        name="fn_add_numbers",
        parameters={
            "a": 2,
            "b": 3,
            "c": 4,
        },
    )

    invalid_type_call = FunctionCallOutput(
        prompt="What is the sum of 2 and 3?",
        name="fn_add_numbers",
        parameters={
            "a": "hello",
            "b": False,
        },
    )

    valid_greet_call = FunctionCallOutput(
        prompt="Greet john",
        name="fn_greet",
        parameters={
            "name": "john",
        },
    )

    validate_function_call(valid_call, definition)
    print("Valid call: OK")

    try:
        validate_function_call(invalid_type_call, definition)
    except FunctionCallValidationError as error:
        print(f"Invalid type: {error}")

    try:
        validate_function_call(missing_parameter_call, definition)
    except FunctionCallValidationError as error:
        print(f"Missing parameter: {error}")

    try:
        validate_function_call(extra_parameter_call, definition)
    except FunctionCallValidationError as error:
        print(f"Extra parameter: {error}")

    try:
        validate_function_call(valid_greet_call, greet_definition)
        print("Valid greet call: OK")
    except FunctionCallValidationError as error:
        print(f"Invalid greet call: {error}")


if __name__ == "__main__":
    main()
