"""Stage 6 — cart state, and the machine that moves through it."""

from __future__ import annotations

from enum import Enum

from voice_order.types import Cart, CartLine, Candidate, OrderIntent


class State(str, Enum):
    GREETING = "greeting"
    LISTENING = "listening"      # waiting for an item
    CONFIRMING = "confirming"    # read back, waiting for yes/no
    CLARIFYING = "clarifying"    # ambiguous, asking a narrowing question
    REVIEWING = "reviewing"      # reading the cart back
    CLOSED = "closed"


class CallState:
    """The state machine. Deliberately explicit rather than prompt-driven —
    a wrong order has to be attributable to a transition, not to a vibe."""

    def __init__(self, call_id: str) -> None:
        raise NotImplementedError("stage 6")

    @property
    def cart(self) -> Cart:
        raise NotImplementedError("stage 6")

    def add_line(self, intent: OrderIntent, candidate: Candidate) -> CartLine:
        raise NotImplementedError("stage 6")

    def confirm_pending(self, yes: bool) -> None:
        raise NotImplementedError("stage 6")

    def transition(self, event: str) -> State:
        raise NotImplementedError("stage 6")
