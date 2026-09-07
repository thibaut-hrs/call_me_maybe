"""Pydantic models representing callable function definitions."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ParameterDefinition(BaseModel):
    """Represent the definition of a function parameter."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["string", "number"]


class ReturnDefinition(BaseModel):
    """Represent the definition of a function return value."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["string", "number"]


class FunctionDefinition(BaseModel):
    """Represent a function available to the language model."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ReturnDefinition
