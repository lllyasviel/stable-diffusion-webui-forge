"""
Pydantic schema adapter for Starlette Request type
This fixes the PydanticSchemaGenerationError for starlette.requests.Request
"""

from typing import Any, Dict
import pydantic.v1 as pydantic_v1
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema
from starlette.requests import Request


# Monkey patch to add Pydantic schema support to Starlette Request
def __get_pydantic_core_schema__(
    cls, 
    source_type: Any, 
    handler: GetCoreSchemaHandler
) -> core_schema.CoreSchema:
    """Custom schema handler for starlette.requests.Request"""
    return core_schema.any_schema()


# Apply the monkey patch
Request.__get_pydantic_core_schema__ = classmethod(__get_pydantic_core_schema__)


# Also add support for Pydantic v1 if needed
def __get_validators__(cls):
    yield cls.validate


def validate(cls, v):
    if isinstance(v, Request):
        return v
    # For non-Request objects, just return as-is or raise validation error
    raise ValueError(f"Expected Request object, got {type(v)}")


Request.__get_validators__ = classmethod(__get_validators__)
Request.validate = classmethod(validate)


# Add schema for Pydantic v1
def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
    field_schema.update(
        type="object",
        description="Starlette Request object",
        properties={},
        additionalProperties=True
    )


Request.__modify_schema__ = classmethod(__modify_schema__)
