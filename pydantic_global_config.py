"""
Global Pydantic configuration to handle starlette Request type
"""

import os
import warnings
from typing import Any, Type

# Suppress Pydantic warnings about schema generation
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Set environment variable to allow arbitrary types globally
os.environ["PYDANTIC_ARBITRARY_TYPES_ALLOWED"] = "true"

try:
    from pydantic import ConfigDict
    from pydantic._internal._config import ConfigWrapper
    from pydantic._internal._generate_schema import GenerateSchema
    from starlette.requests import Request
    
    # Store original method
    _original_unknown_type_schema = GenerateSchema._unknown_type_schema
    
    def _patched_unknown_type_schema(self, obj: Type[Any]) -> Any:
        """Patched version that allows Request type"""
        if obj is Request:
            # Return a schema that accepts any value for Request type
            from pydantic_core import core_schema
            return core_schema.any_schema()
        
        # Fall back to original behavior for other types
        return _original_unknown_type_schema(self, obj)
    
    # Apply the patch
    GenerateSchema._unknown_type_schema = _patched_unknown_type_schema
    
    print("Applied Pydantic schema patch for starlette.requests.Request")
    
except ImportError as e:
    print(f"Could not apply Pydantic patch: {e}")
except Exception as e:
    print(f"Error applying Pydantic patch: {e}")
