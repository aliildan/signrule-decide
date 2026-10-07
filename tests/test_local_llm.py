"""The local-LLM client must refuse anything that would leave the machine."""

from __future__ import annotations

import pytest

from signrule.qa.local_llm import NotLocalError, check_local


def test_refuses_remote_server():
    with pytest.raises(NotLocalError):
        check_local("https://api.example.com", "qwen3.5:35b-a3b-q4_K_M")


def test_refuses_cloud_models():
    with pytest.raises(NotLocalError):
        check_local("http://127.0.0.1:11434", "gpt-oss:120b-cloud")


def test_accepts_local_model():
    check_local("http://127.0.0.1:11434", "qwen3.5:35b-a3b-q4_K_M")
    check_local("http://localhost:11434", "qwen3.5:9b-q4_K_M")
