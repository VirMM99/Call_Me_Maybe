from pydantic import BaseModel, Field, ConfigDict
from enum import Enum
from typing import Any


class ParameterType(str, Enum):
    """Supported parameter types"""
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"


class ParameterDefinition(BaseModel):
    """Validate the type of the parameter"""
    # ConfigDict if there is a extra param 
    # that I did not define, reject it
    model_config = ConfigDict(extra='forbid')
    type: ParameterType


class FunctionDefinitionCheck(BaseModel):
    """Check and validate the availables functions definitions """
    model_config = ConfigDict(extra='forbid')

    name: str = Field(..., min_length=1)
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ParameterDefinition


class PromptItem(BaseModel):
    """What we are gonna sent: prompt"""
    model_config = ConfigDict(extra='forbid')
    prompt: str = Field(..., min_length=1)


class FunctionCallOutput(BaseModel):
    """Check and validate the function call output"""
    model_config = ConfigDict(extra='forbid')
    prompt: str = Field(...)
    name: str = Field(..., min_length=1)
    parameters: dict[str, Any] = Field(...)
