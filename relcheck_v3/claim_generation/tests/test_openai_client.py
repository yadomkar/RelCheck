"""Property-based and unit tests for OpenAIClient.

Property 3: Retry with exponential backoff
Unit tests: missing API key, env var fallback, custom model ID

Validates: Requirements 1.5, 9.2, 9.3, 9.5, 12.3
"""

import os
from unittest.mock import MagicMock, patch

import openai
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from relcheck_v3.claim_generation.openai_client import OpenAIClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_create(n_failures: int):
    """Return a mock chat.completions.create that fails n_failures times then succeeds."""
    call_count = 0

    def _side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= n_failures:
            raise openai.APIError(
                message=f"transient error #{call_count}",
                request=MagicMock(),
                body=None,
            )
        # Success response
        choice = MagicMock()
        choice.message.content = "success_response"
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    return _side_effect


def _build_client_with_mocks(n_failures: int):
    """Build an OpenAIClient with a mocked openai.OpenAI and mocked time.sleep.

    Returns (client, sleep_mock) so tests can inspect sleep calls.
    """
    with patch("relcheck_v3.claim_generation.openai_client.openai.OpenAI") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.chat.completions.create.side_effect = _make_mock_create(n_failures)
        mock_cls.return_value = mock_instance

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
            client = OpenAIClient()

    # Re-attach the mock so calls go through it
    client._client = mock_instance

    sleep_mock = MagicMock()
    return client, sleep_mock


# ---------------------------------------------------------------------------
# Property 3: Retry with exponential backoff
# ---------------------------------------------------------------------------
# Feature: woodpecker-correction, Property 3: Retry with exponential backoff


@given(n_failures=st.integers(min_value=0, max_value=5))
@settings(max_examples=100)
def test_retry_with_exponential_backoff(n_failures: int) -> None:
    """For N ≤ 3 consecutive transient errors followed by success, client returns
    the successful response. For >3 consecutive errors, client raises.
    Delay between retry i and i+1 is at least 2^i seconds.

    **Validates: Requirements 1.5, 9.3**
    """
    client, _ = _build_client_with_mocks(n_failures)

    with patch("relcheck_v3.claim_generation.openai_client.time.sleep") as sleep_mock:
        if n_failures <= 3:
            # Should succeed — initial attempt + up to 3 retries covers ≤3 failures
            result = client.chat("system", "user")
            assert result == "success_response"

            # Verify exponential backoff delays
            assert sleep_mock.call_count == n_failures
            for i, call in enumerate(sleep_mock.call_args_list):
                delay = call[0][0]
                expected_min = 2**i  # 1, 2, 4 seconds (base_delay=1, so 2^0*1, 2^1*1, 2^2*1)
                assert delay >= expected_min, (
                    f"Retry {i}: delay {delay} < expected minimum {expected_min}"
                )
        else:
            # >3 failures means all 4 attempts (initial + 3 retries) fail → raises
            with pytest.raises(openai.APIError):
                client.chat("system", "user")

            # Should have slept 3 times (between attempts 0→1, 1→2, 2→3)
            assert sleep_mock.call_count == 3
            for i, call in enumerate(sleep_mock.call_args_list):
                delay = call[0][0]
                expected_min = 2**i
                assert delay >= expected_min


# ---------------------------------------------------------------------------
# Unit Tests (Task 3.3)
# ---------------------------------------------------------------------------


class TestOpenAIClientMissingKey:
    """Missing API key (no arg, no env var) raises ValueError mentioning OPENAI_API_KEY.

    **Validates: Requirements 9.5**
    """

    def test_missing_api_key_raises_value_error(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with patch("relcheck_v3.claim_generation.openai_client.openai.OpenAI"):
                with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                    OpenAIClient(api_key="")


class TestOpenAIClientEnvVar:
    """API key from env var works when constructor arg is empty.

    **Validates: Requirements 9.2**
    """

    def test_env_var_fallback(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-test-key"}, clear=False):
            with patch("relcheck_v3.claim_generation.openai_client.openai.OpenAI") as mock_cls:
                client = OpenAIClient(api_key="")
                # Should have initialized the openai client with the env var key
                mock_cls.assert_called_once_with(api_key="env-test-key")


class TestOpenAIClientCustomModel:
    """Custom model ID is passed through to the openai SDK call.

    **Validates: Requirements 12.3**
    """

    def test_custom_model_passed_to_sdk(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
            with patch("relcheck_v3.claim_generation.openai_client.openai.OpenAI") as mock_cls:
                mock_instance = MagicMock()
                choice = MagicMock()
                choice.message.content = "hello"
                resp = MagicMock()
                resp.choices = [choice]
                mock_instance.chat.completions.create.return_value = resp
                mock_cls.return_value = mock_instance

                custom_model = "gpt-4o-2024-05-13"
                client = OpenAIClient(api_key="test-key", model=custom_model)
                result = client.chat("sys", "usr")

                # Verify the custom model was passed to the SDK
                call_kwargs = mock_instance.chat.completions.create.call_args
                assert call_kwargs.kwargs["model"] == custom_model
                assert result == "hello"
