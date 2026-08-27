"""Does this model actually drive the agent?

Tool-calling is not a yes/no capability. A model can emit perfectly valid
calls and still sequence them wrongly -- adding before searching, placing
before reading a total -- and only a scripted conversation shows that.

So this runs one fixed conversation and checks specific behaviours rather than
eyeballing the transcript. It exists because "qwen2.5:1.5b hallucinated a
product, 3b invented an id, and both looked fine in isolation" is not a thing
anyone should have to rediscover for themselves.

    voice-order check-model
    voice-order check-model --model qwen2.5:7b-instruct

The checks are deliberately behavioural, not textual. Nothing here greps the
reply for a phrase -- every one of them looks at what the model *did* to the
cart, because that is the part that can go wrong expensively.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# A short order, with the two moments that trip models up: a vague product
# name, and a "yes" that means "place it" rather than "add it".
SCRIPT = [
    "I need two ACDelco spark plugs",
    "yes that one",
    "no that's everything",
    "yes go ahead",
]


@dataclass
class Result:
    model: str
    turns: list[dict] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for v in self.checks.values() if v)

    @property
    def total(self) -> int:
        return len(self.checks)


def run(model: str | None = None, script: list[str] | None = None) -> Result:
    """Talk to the model, then judge what it did rather than what it said."""
    from voice_order.agent.brain import Brain
    from voice_order.agent.state import OrderSession
    from voice_order.llm.client import OpenAICompatClient, from_config
    from voice_order.retrieval.fusion import Retriever

    client = from_config()
    if model:
        client = OpenAICompatClient(
            base_url=client.base_url, model=model, api_key=client.api_key
        )

    session = OrderSession()
    session._retriever = Retriever.load(retrievers="lexical")
    brain = Brain(client, session)
    result = Result(model=client.model)

    for said in script or SCRIPT:
        started = time.perf_counter()
        try:
            reply = brain.say(said)
        except Exception as exc:  # a model that cannot be reached is a result
            result.notes.append(f"failed on {said!r}: {type(exc).__name__}: {exc}")
            break
        result.turns.append(
            {
                "said": said,
                "reply": reply,
                "seconds": round(time.perf_counter() - started, 1),
                "cart": session._snapshot(),
            }
        )

    calls = [t for turn in brain.tool_log for t in turn.get("tools", [])]
    names = [c["name"] for c in calls]
    errors = [c for c in calls if isinstance(c["result"], dict) and "error" in c["result"]]

    def first(name: str) -> int:
        return names.index(name) if name in names else 10**6

    result.checks = {
        "used the tools at all": bool(calls),
        "searched before adding": first("search_products") < first("add_to_cart"),
        "never invented a product id": not any(
            "no product" in c["result"].get("error", "") for c in errors
        ),
        "read the cart before placing": first("read_cart") < first("place_order"),
        "placed the order": "place_order" in names
        and not any(
            c["name"] == "place_order" and "error" in c["result"] for c in calls
        ),
        "something ended up in the cart": bool(session.lines),
        "replies short enough to speak": all(
            len(t["reply"]) < 220 for t in result.turns
        ),
    }

    seconds = [t["seconds"] for t in result.turns]
    if seconds:
        result.notes.append(
            f"{sum(seconds)/len(seconds):.0f}s per turn on average, "
            f"slowest {max(seconds):.0f}s"
        )
    if errors:
        result.notes.append(
            "tool errors: " + "; ".join(sorted({c["result"]["error"][:60] for c in errors}))
        )
    result.notes.append("tool calls: " + " -> ".join(names) if names else "no tool calls")
    return result


def report(result: Result) -> None:
    print()
    print(f"model: {result.model}")
    print()
    for said, turn in zip([t["said"] for t in result.turns], result.turns, strict=True):
        print(f"  you   {said}")
        print(f"  agent {turn['reply'][:150]}")
        cart = turn["cart"]
        print(f"        [{turn['seconds']}s]  cart {cart['item_count']} item(s)  "
              f"total {cart['total']}")
        print()

    width = max(len(k) for k in result.checks) + 2 if result.checks else 20
    for name, ok in result.checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}")
    print()
    print(f"  {result.passed}/{result.total} checks passed")
    for note in result.notes:
        print(f"  - {note}")
