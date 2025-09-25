"""Test for parse:error hook fix when instructor.ValidationError is raised in JSON mode.

This test reproduces the bug reported in GitHub issue #1817 where the parse:error
hook was not emitted when instructor.exceptions.ValidationError was raised by the
JSON parser but not caught in retry_async/retry_sync exception handlers.
"""

import pytest
from unittest.mock import Mock, patch
from pydantic import BaseModel, Field
from typing import List

import instructor
from instructor.core.exceptions import ValidationError as InstructorValidationError
from instructor.core.retry import retry_sync, retry_async
from instructor.mode import Mode
from instructor.core.hooks import Hooks


class SimpleOutput(BaseModel):
    name: str = Field(..., description="Required name")
    count: int = Field(..., description="Required count") 
    items: List[str] = Field(..., description="Required items list")


def test_parse_error_hook_emitted_for_instructor_validation_error_sync():
    """Test that parse:error hook is emitted when InstructorValidationError is raised in sync retry."""
    # Track if parse error hook was called
    parse_error_called = []
    
    def on_parse_error(error):
        parse_error_called.append(type(error).__name__)
    
    # Set up hooks
    hooks = Hooks()
    hooks.on("parse:error", on_parse_error)
    
    # Mock function that raises InstructorValidationError
    def mock_func(*args, **kwargs):
        # Return a mock response first
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "{}"  # Empty JSON that will fail validation
        return mock_response
    
    # Mock process_response to raise InstructorValidationError
    with patch('instructor.core.retry.process_response') as mock_process:
        mock_process.side_effect = InstructorValidationError("Validation failed")
        
        # Call retry_sync and expect it to fail with InstructorRetryException
        with pytest.raises(Exception):  # This should raise InstructorRetryException eventually
            retry_sync(
                func=mock_func,
                response_model=SimpleOutput,
                args=(),
                kwargs={"messages": [{"role": "user", "content": "test"}]},
                max_retries=1,
                mode=Mode.JSON,
                hooks=hooks
            )
    
    # Verify that parse:error hook was called with InstructorValidationError
    assert len(parse_error_called) == 1
    assert parse_error_called[0] == "ValidationError"


@pytest.mark.asyncio
async def test_parse_error_hook_emitted_for_instructor_validation_error_async():
    """Test that parse:error hook is emitted when InstructorValidationError is raised in async retry."""
    # Track if parse error hook was called
    parse_error_called = []
    
    def on_parse_error(error):
        parse_error_called.append(type(error).__name__)
    
    # Set up hooks
    hooks = Hooks()
    hooks.on("parse:error", on_parse_error)
    
    # Mock async function that raises InstructorValidationError
    async def mock_async_func(*args, **kwargs):
        # Return a mock response first
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "{}"  # Empty JSON that will fail validation
        return mock_response
    
    # Mock process_response_async to raise InstructorValidationError
    with patch('instructor.core.retry.process_response_async') as mock_process:
        mock_process.side_effect = InstructorValidationError("Validation failed")
        
        # Call retry_async and expect it to fail with InstructorRetryException
        with pytest.raises(Exception):  # This should raise InstructorRetryException eventually
            await retry_async(
                func=mock_async_func,
                response_model=SimpleOutput,
                args=(),
                kwargs={"messages": [{"role": "user", "content": "test"}]},
                max_retries=1,
                mode=Mode.JSON,
                hooks=hooks
            )
    
    # Verify that parse:error hook was called with InstructorValidationError
    assert len(parse_error_called) == 1
    assert parse_error_called[0] == "ValidationError"


def test_parse_error_hook_still_works_for_pydantic_validation_error_sync():
    """Test that parse:error hook still works for pydantic.ValidationError (regression test)."""
    from pydantic import ValidationError as PydanticValidationError
    
    # Track if parse error hook was called
    parse_error_called = []
    
    def on_parse_error(error):
        parse_error_called.append(type(error).__name__)
    
    # Set up hooks
    hooks = Hooks()
    hooks.on("parse:error", on_parse_error)
    
    # Mock function
    def mock_func(*args, **kwargs):
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "{}"
        return mock_response
    
    # Mock process_response to raise pydantic ValidationError
    with patch('instructor.core.retry.process_response') as mock_process:
        # Create a simple pydantic ValidationError  
        try:
            SimpleOutput(name="test", count="invalid", items=[])  # This will raise ValidationError
        except PydanticValidationError as e:
            mock_process.side_effect = e
        
        # Call retry_sync and expect it to fail
        with pytest.raises(Exception):
            retry_sync(
                func=mock_func,
                response_model=SimpleOutput,
                args=(),
                kwargs={"messages": [{"role": "user", "content": "test"}]},
                max_retries=1,
                mode=Mode.JSON,
                hooks=hooks
            )
    
    # Verify that parse:error hook was called with pydantic ValidationError
    assert len(parse_error_called) == 1
    assert parse_error_called[0] == "ValidationError"


@pytest.mark.asyncio
async def test_parse_error_hook_still_works_for_pydantic_validation_error_async():
    """Test that parse:error hook still works for pydantic.ValidationError in async (regression test)."""
    from pydantic import ValidationError as PydanticValidationError
    
    # Track if parse error hook was called
    parse_error_called = []
    
    def on_parse_error(error):
        parse_error_called.append(type(error).__name__)
    
    # Set up hooks
    hooks = Hooks()
    hooks.on("parse:error", on_parse_error)
    
    # Mock async function
    async def mock_async_func(*args, **kwargs):
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "{}"
        return mock_response
    
    # Mock process_response_async to raise pydantic ValidationError
    with patch('instructor.core.retry.process_response_async') as mock_process:
        # Create a simple pydantic ValidationError  
        try:
            SimpleOutput(name="test", count="invalid", items=[])  # This will raise ValidationError
        except PydanticValidationError as e:
            mock_process.side_effect = e
        
        # Call retry_async and expect it to fail
        with pytest.raises(Exception):
            await retry_async(
                func=mock_async_func,
                response_model=SimpleOutput,
                args=(),
                kwargs={"messages": [{"role": "user", "content": "test"}]},
                max_retries=1,
                mode=Mode.JSON,
                hooks=hooks
            )
    
    # Verify that parse:error hook was called with pydantic ValidationError
    assert len(parse_error_called) == 1
    assert parse_error_called[0] == "ValidationError"


def test_parse_error_hook_emitted_for_json_decode_error_sync():
    """Test that parse:error hook is emitted for JSONDecodeError (regression test)."""
    from json import JSONDecodeError
    
    # Track if parse error hook was called
    parse_error_called = []
    
    def on_parse_error(error):
        parse_error_called.append(type(error).__name__)
    
    # Set up hooks
    hooks = Hooks()
    hooks.on("parse:error", on_parse_error)
    
    # Mock function
    def mock_func(*args, **kwargs):
        mock_response = Mock()
        return mock_response
    
    # Mock process_response to raise JSONDecodeError
    with patch('instructor.core.retry.process_response') as mock_process:
        mock_process.side_effect = JSONDecodeError("Invalid JSON", "test", 0)
        
        # Call retry_sync and expect it to fail
        with pytest.raises(Exception):
            retry_sync(
                func=mock_func,
                response_model=SimpleOutput,
                args=(),
                kwargs={"messages": [{"role": "user", "content": "test"}]},
                max_retries=1,
                mode=Mode.JSON,
                hooks=hooks
            )
    
    # Verify that parse:error hook was called with JSONDecodeError
    assert len(parse_error_called) == 1
    assert parse_error_called[0] == "JSONDecodeError"


def test_all_parse_errors_caught_in_same_handler():
    """Test that both instructor.ValidationError and pydantic.ValidationError are caught by the same handler."""
    from pydantic import ValidationError as PydanticValidationError
    from json import JSONDecodeError
    
    # Track all parse errors
    parse_errors = []
    
    def on_parse_error(error):
        parse_errors.append((type(error).__name__, str(error)))
    
    # Set up hooks
    hooks = Hooks()
    hooks.on("parse:error", on_parse_error)
    
    def mock_func(*args, **kwargs):
        mock_response = Mock()
        return mock_response
    
    # Test InstructorValidationError
    with patch('instructor.core.retry.process_response') as mock_process:
        mock_process.side_effect = InstructorValidationError("Instructor validation failed")
        
        with pytest.raises(Exception):
            retry_sync(
                func=mock_func,
                response_model=SimpleOutput,
                args=(),
                kwargs={"messages": [{"role": "user", "content": "test"}]},
                max_retries=1,
                mode=Mode.JSON,
                hooks=hooks
            )
    
    # Test PydanticValidationError
    with patch('instructor.core.retry.process_response') as mock_process:
        # Create a simple pydantic ValidationError  
        try:
            SimpleOutput(name="test", count="invalid", items=[])  # This will raise ValidationError
        except PydanticValidationError as e:
            mock_process.side_effect = e
        
        with pytest.raises(Exception):
            retry_sync(
                func=mock_func,
                response_model=SimpleOutput,
                args=(),
                kwargs={"messages": [{"role": "user", "content": "test"}]},
                max_retries=1,
                mode=Mode.JSON,
                hooks=hooks
            )
    
    # Test JSONDecodeError
    with patch('instructor.core.retry.process_response') as mock_process:
        mock_process.side_effect = JSONDecodeError("Invalid JSON", "test", 0)
        
        with pytest.raises(Exception):
            retry_sync(
                func=mock_func,
                response_model=SimpleOutput,
                args=(),
                kwargs={"messages": [{"role": "user", "content": "test"}]},
                max_retries=1,
                mode=Mode.JSON,
                hooks=hooks
            )
    
    # Verify all three error types were caught and emitted through parse:error hook
    assert len(parse_errors) == 3
    error_types = [error[0] for error in parse_errors]
    assert "ValidationError" in error_types  # This could be either instructor or pydantic
    assert "JSONDecodeError" in error_types
    
    # Verify error messages were preserved
    error_messages = [error[1] for error in parse_errors]
    assert any("Instructor validation failed" in msg for msg in error_messages)
    assert any("validation error" in msg.lower() for msg in error_messages)  # Pydantic errors contain validation error
    assert any("Invalid JSON" in msg for msg in error_messages)