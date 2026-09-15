from pydantic import (
    Basemodel, Field, field_validator, ConfigDict, ValidationError)
from typing import Any


class ValidateParameters(Basemodel):
    """Check and validate the function parameters"""
    model_config = ConfigDict(extra='forbid')
    type: str

class FunctionDefinitionCheck(Basemodel):
    """Check and validate the funtions availables"""
    model_config = ConfigDict(extra='forbid')
    name: str = Field(..., min_length=4)
    description: str = Field(...)
    parameters: dict[str, ValidateParameters] = Field(...)
    returns: ValidateParameters


class PromptItem(Basemodel):
    model_config = ConfigDict(extra='forbid')
    prompt: str = Field(..., min_lenght=1)


class FuctionCallOutput(Basemodel):
    """Check and validate funxtion call output"""
    model_config = ConfigDict(extra='forbid')
    prompt: str = Field(...)
    name: str = Field(..., min_length=4)
    parameters: dict[str, Any] = Field(...)


def check_basemodel():
    p_type = PType.STR
    parameter_dict = {
        "name": {
            "type": p_type
        }
    }
    r_type = PType.STR
    return_dict = {"type": r_type}
    function_def_dict = {
        "name": "fn_greet",
        "description": "Generate a greeting message for a person by name.",
        "parameters": parameter_dict,
        "returns": return_dict
    }
    try:
        function_basemodel = FunctionDefinitionCheck(**function_def_dict)
        function_definition = {
            "name": function_basemodel.name,
            "description": function_basemodel.description,
            "parameters": function_basemodel.parameters,
            "returns": function_basemodel.returns
        }
        print(function_definition)
    except ValidationError as e:
        error_msg = e.errors()[0]["msg"]
        raise ValueError(f"An error ocurred: {error_msg}")


if __name__ == "__main__":
    check_basemodel()