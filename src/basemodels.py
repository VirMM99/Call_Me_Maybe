from pydantic import BaseModel, Field, ConfigDict
from typing import Any


class ParameterDefinition(BaseModel):
    """Check and validate the function parameters types"""
    model_config = ConfigDict(extra='forbid')
    type: str


class FunctionDefinitionCheck(BaseModel):
    """Check and validate the availables functions definitions """
    model_config = ConfigDict(extra='forbid')
    name: str = Field(..., min_length=1)
    description: str = Field(...)
    parameters: dict[str, ParameterDefinition] = Field(...)
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
