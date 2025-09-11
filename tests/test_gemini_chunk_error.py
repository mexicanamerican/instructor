"""Tests for Gemini chunk.text error handling in partial.py"""

import pytest
from instructor.dsl.partial import PartialBase
from instructor.mode import Mode


class MockChunk:
    """Mock chunk that raises ValueError when accessing .text"""

    @property
    def text(self):
        raise ValueError(
            "Invalid operation: The `response.text` quick accessor requires the response to contain a valid `Part`, but none were returned. The candidate's [finish_reason](https://ai.google.dev/api/generate-content#finishreason) is 1."
        )


class MockValidChunk:
    """Mock chunk with valid text"""

    @property
    def text(self):
        return '{"key": "value"}'


def test_extract_json_handles_gemini_json_invalid_part():
    """Test that extract_json gracefully handles chunks with invalid Parts for GEMINI_JSON mode"""
    # Create mock chunks - one invalid, one valid
    invalid_chunk = MockChunk()
    valid_chunk = MockValidChunk()
    completion = [invalid_chunk, valid_chunk]

    # Extract JSON chunks
    json_chunks = list(PartialBase.extract_json(completion, Mode.GEMINI_JSON))

    # Should only get the valid chunk's text, invalid chunk should be skipped
    assert len(json_chunks) == 1
    assert json_chunks[0] == '{"key": "value"}'


def test_extract_json_handles_genai_structured_outputs_invalid_part():
    """Test that extract_json gracefully handles chunks with invalid Parts for GENAI_STRUCTURED_OUTPUTS mode"""
    # Create mock chunks - one invalid, one valid
    invalid_chunk = MockChunk()
    valid_chunk = MockValidChunk()
    completion = [invalid_chunk, valid_chunk]

    # Extract JSON chunks
    json_chunks = list(
        PartialBase.extract_json(completion, Mode.GENAI_STRUCTURED_OUTPUTS)
    )

    # Should only get the valid chunk's text, invalid chunk should be skipped
    assert len(json_chunks) == 1
    assert json_chunks[0] == '{"key": "value"}'


def test_extract_json_handles_all_invalid_chunks():
    """Test that extract_json handles when all chunks have invalid Parts"""
    # Create only invalid chunks
    invalid_chunk1 = MockChunk()
    invalid_chunk2 = MockChunk()
    completion = [invalid_chunk1, invalid_chunk2]

    # Extract JSON chunks
    json_chunks = list(PartialBase.extract_json(completion, Mode.GEMINI_JSON))

    # Should get empty list when all chunks are invalid
    assert len(json_chunks) == 0


def test_extract_json_reraises_other_valueerrors():
    """Test that extract_json re-raises ValueErrors that aren't about invalid Parts"""

    class MockChunkOtherError:
        @property
        def text(self):
            raise ValueError("Some other error message")

    other_error_chunk = MockChunkOtherError()
    completion = [other_error_chunk]

    # Should re-raise the ValueError since it's not about invalid Parts
    with pytest.raises(ValueError, match="Some other error message"):
        list(PartialBase.extract_json(completion, Mode.GEMINI_JSON))


@pytest.mark.asyncio
async def test_extract_json_async_handles_gemini_json_invalid_part():
    """Test that extract_json_async gracefully handles chunks with invalid Parts for GEMINI_JSON mode"""

    async def async_completion():
        yield MockChunk()  # Invalid chunk
        yield MockValidChunk()  # Valid chunk

    # Extract JSON chunks
    json_chunks = []
    async for chunk in PartialBase.extract_json_async(
        async_completion(), Mode.GEMINI_JSON
    ):
        json_chunks.append(chunk)

    # Should only get the valid chunk's text, invalid chunk should be skipped
    assert len(json_chunks) == 1
    assert json_chunks[0] == '{"key": "value"}'


@pytest.mark.asyncio
async def test_extract_json_async_handles_genai_structured_outputs_invalid_part():
    """Test that extract_json_async gracefully handles chunks with invalid Parts for GENAI_STRUCTURED_OUTPUTS mode"""

    async def async_completion():
        yield MockChunk()  # Invalid chunk
        yield MockValidChunk()  # Valid chunk

    # Extract JSON chunks
    json_chunks = []
    async for chunk in PartialBase.extract_json_async(
        async_completion(), Mode.GENAI_STRUCTURED_OUTPUTS
    ):
        json_chunks.append(chunk)

    # Should only get the valid chunk's text, invalid chunk should be skipped
    assert len(json_chunks) == 1
    assert json_chunks[0] == '{"key": "value"}'
