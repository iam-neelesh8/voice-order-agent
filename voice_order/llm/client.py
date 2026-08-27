"""Stage 6 -- one interface to any chat model, local or hosted.

Ollama, vLLM, llama.cpp's server and LM Studio all expose the same
OpenAI-compatible `/v1/chat/completions` endpoint, and so do the commercial
APIs. So there is one client and switching backends is two lines of config:

    llm:
      base_url: http://localhost:11434/v1     # Ollama
      model: qwen2.5:7b-instruct
      api_key_env: null

Written on stdlib urllib rather than the `openai` package. The request is a
POST with a JSON body; a dependency to build that would be a dependency to
pin, upgrade and explain, and this project already avoids one for the catalog
download.

`FakeClient` is not a testing afterthought -- it is what makes the entire
conversation layer testable without a model running. Every agent test scripts
its replies, so the cart, the thresholds and the tool wiring are verified
deterministically and in milliseconds.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    """A function the model wants run. It cannot run anything itself."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Reply:
    """What came back: something to say, tool calls to make, or both."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMClient(Protocol):
    """The whole surface the agent depends on."""

    def chat(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> Reply: ...


# ---------------------------------------------------------------------------


class OpenAICompatClient:
    """Talks to anything speaking the OpenAI chat-completions shape."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        # Zero by default. A phone order is not a place for creative variance,
        # and it makes a failing conversation reproducible.
        self.temperature = temperature
        self.timeout = timeout

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> Reply:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            raise RuntimeError(f"LLM returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"cannot reach the model at {self.base_url} ({exc.reason}). "
                "Is Ollama running? `ollama serve`, then `ollama pull "
                f"{self.model}`."
            ) from exc

        return self._parse(body)

    @staticmethod
    def _parse(body: dict) -> Reply:
        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError(f"model returned no choices: {str(body)[:300]}")
        message = choices[0].get("message") or {}

        calls: list[ToolCall] = []
        for i, raw in enumerate(message.get("tool_calls") or []):
            function = raw.get("function") or {}
            arguments = function.get("arguments")
            # Servers disagree here: some send a JSON string, some a dict.
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments or "{}")
                except json.JSONDecodeError:
                    # A model that emits malformed arguments is a normal
                    # failure, not an exception -- the agent turns it into a
                    # clarifying question rather than crashing the call.
                    arguments = {"_malformed": arguments}
            calls.append(
                ToolCall(
                    id=str(raw.get("id") or f"call_{i}"),
                    name=str(function.get("name") or ""),
                    arguments=arguments if isinstance(arguments, dict) else {},
                )
            )

        return Reply(text=(message.get("content") or "").strip(), tool_calls=calls)


class FakeClient:
    """A scripted client, for tests and for exercising the loop offline.

    Hand it a list of `Reply` objects and it returns them in order. Every
    message it was asked about is kept in `seen`, so a test can assert what the
    agent actually told the model.
    """

    def __init__(self, replies: list[Reply]) -> None:
        self.replies = list(replies)
        self.seen: list[list[dict]] = []
        self.tools_offered: list[dict] | None = None

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> Reply:
        self.seen.append([dict(m) for m in messages])
        self.tools_offered = tools
        if not self.replies:
            raise AssertionError(
                "FakeClient ran out of scripted replies -- the agent asked the "
                "model more times than the test expected"
            )
        return self.replies.pop(0)


# ---------------------------------------------------------------------------


def from_config() -> LLMClient:
    """Build the client described by configs/agent.yaml."""
    from voice_order import config

    cfg = config.load("agent")
    base_url = str(cfg.get("llm.base_url", "http://localhost:11434/v1"))
    model = str(cfg.get("llm.model", "qwen2.5:7b-instruct"))

    key_env = cfg.get("llm.api_key_env", None)
    api_key = os.environ.get(str(key_env)) if key_env else None
    if key_env and not api_key:
        raise RuntimeError(
            f"configs/agent.yaml expects the API key in ${key_env}, which is unset"
        )

    return OpenAICompatClient(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=float(cfg.get("llm.temperature", 0.0)),
        timeout=float(cfg.get("llm.timeout_s", 120)),
    )
