"""Stage 6 -- the conversation, driven by a model but not owned by one.

One turn is: give the model what the caller said, let it ask for tools, run
them, hand back the results, repeat until it has something to say. The model
never touches the cart; `llm/tools.py` and `agent/state.py` do, and they
refuse anything invalid.

Two limits worth naming, because both are failure modes rather than
theoretical concerns:

  * `MAX_TOOL_ROUNDS` stops a model looping on a tool that keeps returning an
    error. Without it a confused model can hold a phone line open forever.
  * a model that returns neither text nor a tool call gets one nudge, then a
    fixed fallback line. Silence on a phone call is worse than a clumsy
    sentence.
"""

from __future__ import annotations

import json

from voice_order.llm import tools as tool_defs
from voice_order.llm.client import LLMClient, Reply

MAX_TOOL_ROUNDS = 6

SYSTEM_PROMPT = """You answer the phone for a parts shop and take orders.

You are SPEAKING, not writing. Every reply must be one or two short sentences.
Never write a list. Never use numbers, bullets or line breaks. If you would
normally list options, name at most two, in a sentence.

Use short product names. Say "the ACDelco spark plug", not the full catalog
title -- those are written for a web page and sound absurd read aloud.

WHAT TO DO
- Caller names something -> call search_products.
- The result has a `guidance` field. Do exactly what it says.
- Never invent a product, a price, or a total. If you did not get it from a
  tool, you do not know it.
- Caller says they are done -> call read_cart, then say the items and the
  total, then stop and wait.
- Caller agrees to that total ("yes", "go ahead", "that's right") -> call
  place_order.
- Caller says yes to a product you read back -> call add_to_cart.

The difference matters: "yes" after you read back a PRODUCT means add it.
"yes" after you read back the TOTAL means place the order.

If you did not understand, say so and ask for the brand or the part number.
Do not guess.
"""

FALLBACK_REPLY = "Sorry, could you say that again?"


def _tool_result_message(call_id: str, name: str, result: dict) -> dict:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": json.dumps(result, ensure_ascii=False)[:4000],
    }


class Brain:
    """Runs one caller utterance to one spoken reply.

    Holds the message history for the call, so "actually make that three"
    resolves against what was said earlier without any of it being re-derived.
    """

    def __init__(self, client: LLMClient, session, system_prompt: str | None = None) -> None:
        self.client = client
        self.session = session
        self.messages: list[dict] = [
            {"role": "system", "content": system_prompt or SYSTEM_PROMPT}
        ]
        self.tool_log: list[dict] = []

    def say(self, utterance: str) -> str:
        """Caller said this. Return what the agent says back."""
        self.messages.append({"role": "user", "content": utterance})
        return self._run()

    def _run(self) -> str:
        turn_tools: list[dict] = []

        for _ in range(MAX_TOOL_ROUNDS):
            reply = self.client.chat(self.messages, tools=tool_defs.SCHEMAS)

            if not reply.wants_tools:
                text = reply.text.strip()
                if text:
                    self.messages.append({"role": "assistant", "content": text})
                    self.tool_log.append({"tools": turn_tools})
                    return text
                break  # nothing said and nothing asked for -- nudge below

            self.messages.append(
                {
                    "role": "assistant",
                    "content": reply.text or None,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                        for call in reply.tool_calls
                    ],
                }
            )

            for call in reply.tool_calls:
                result = tool_defs.execute(call.name, call.arguments, self.session)
                turn_tools.append(
                    {"name": call.name, "arguments": call.arguments, "result": result}
                )
                self.messages.append(_tool_result_message(call.id, call.name, result))

        # Either the model ran out of rounds or said nothing at all. One nudge,
        # then a fixed line -- an open phone line with silence on it is worse
        # than an inelegant sentence.
        self.messages.append(
            {
                "role": "user",
                "content": (
                    "Reply to the caller now, in one short sentence. Do not call "
                    "any more tools."
                ),
            }
        )
        try:
            final = self.client.chat(self.messages, tools=None)
            text = final.text.strip() or FALLBACK_REPLY
        except Exception:
            text = FALLBACK_REPLY

        self.messages.append({"role": "assistant", "content": text})
        self.tool_log.append({"tools": turn_tools, "forced_reply": True})
        return text

    def greeting(self) -> str:
        """The first thing the caller hears. Fixed, not generated.

        There is nothing for a model to decide here, and a generated greeting
        is one more thing that can come out wrong on the first second of a call.
        """
        return "Parts shop, what can I get you?"


def scripted_reply(text: str = "", calls: list[tuple[str, dict]] | None = None) -> Reply:
    """Build a `Reply` for tests without hand-writing tool-call plumbing."""
    from voice_order.llm.client import ToolCall

    return Reply(
        text=text,
        tool_calls=[
            ToolCall(id=f"c{i}", name=name, arguments=args)
            for i, (name, args) in enumerate(calls or [])
        ],
    )
