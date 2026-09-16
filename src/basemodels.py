from pydantic import BaseModel, Field, ConfigDict
from enum import Enum
from typing import Any


class ParameterterType(str, Enum):
    """Supported parameter types"""
    STRING = "string"
    NUMBER = "number"
    INGEGER = " integer"


class ParameterDefinition(BaseModel):
    """Check and validate a parameter definition"""
    model_config = ConfigDict(extra='forbid')
    type: ParameterterType


class FunctionDefinitionCheck(BaseModel):
    """Check and validate the availables functions definitions """
    model_config = ConfigDict(extra='forbid')

    name: str = Field(..., min_length=1)
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ParameterDefinition


class PromptItem(BaseModel):
    model_config = ConfigDict(extra='forbid')
    prompt: str = Field(..., min_length=1)


class FunctionCallOutput(BaseModel):
    """Check and validate function call output"""
    model_config = ConfigDict(extra='forbid')
    prompt: str = Field(...)
    name: str = Field(..., min_length=1)
    parameters: dict[str, Any] = Field(...)
