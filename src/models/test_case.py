"""Pydantic model representing a function calling test case."""

from pydantic import BaseModel, ConfigDict


class TestCase(BaseModel):
    """Represent a prompt used to test function calling."""

    model_config = ConfigDict(extra="forbid")

    prompt: str