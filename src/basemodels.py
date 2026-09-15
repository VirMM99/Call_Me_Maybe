from pydantic import (
    BaseModel, Field, ConfigDict, ValidationError)
from typing import Any


class ParameterDefinition(BaseModel):
    """Check and validate the function parameters"""
    model_config = ConfigDict(extra='forbid')
    type: Any

class FunctionDefinitionCheck(BaseModel):
    """Check and validate the funtions availables"""
    model_config = ConfigDict(extra='forbid')
    name: str = Field(..., min_length=4)
    description: str = Field(...)
    parameters: dict[str, ParameterDefinition] = Field(...)
    returns: ParameterDefinition


class PromptItem(BaseModel):
    model_config = ConfigDict(extra='forbid')
    prompt: str = Field(..., min_length=1)


class FunctionCallOutput(BaseModel):
    """Check and validate funxtion call output"""
    model_config = ConfigDict(extra='forbid')
    prompt: str = Field(...)
    name: str = Field(..., min_length=4)
    parameters: dict[str, Any] = Field(...)
