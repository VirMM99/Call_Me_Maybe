from typing import Any
from .basemodels import (
    FunctionCallOutput,
    FunctionDefinitionCheck
)

# Checking if function exists and are equal
# the parameters are the expected ones
# and the type are the same as the definition


class FunctionCallValidationError(Exception):
    """Raised when a function call
        does not match its definition
    """
    pass


def is_valid_parameter_type(
                        value: Any,
                        expected_type: str,
                        ) -> bool:
    """Check whether a value matches the expected parameter type
    Args:
        value: Parameter value to check
        expected_type: Expected type from the function definition
    Return:
        True if the value matches the expected type,
        otherwise false
    """
    if expected_type == "number":
        return (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                )
    if expected_type == "string":
        return isinstance(value, str)
    return False


def validate_function_call(
    call: FunctionCallOutput,
    definition: FunctionDefinitionCheck,
        ) -> None:
    """Validate a function call against a function definition
    Args:
        call: Function call to validate
        definition: Expected function definition
    Raises:
        FunctionCallValidationError: If the call does not match
            the function definition
    """
    if call.name != definition.name:
        raise FunctionCallValidationError(
            f"Unknown function: '{call.name}'"
        )
    # We use set not to see if it is well sorted
    # but if it has the same parameters
    expected_parameters = set(definition.parameters.keys())
    provided_parameters = set(call.parameters.keys())
    if expected_parameters != provided_parameters:
        raise FunctionCallValidationError(
            "Function parameters do not match the definition"
        )

    for parameter_name, parameter_definition in definition.parameters.items():
        value = call.parameters[parameter_name]

        if not is_valid_parameter_type(
            value,
            parameter_definition.type.value
        ):
            raise FunctionCallValidationError(
                f"Invalid type for parameter '{parameter_name}'"
            )
