"""Stage 6 -- one call, start to hang-up.

Typed input first, deliberately. The whole conversation -- search, cart,
confirmation, total, placing the order -- is exercised by typing, with no
microphone, no ASR and no TTS in the way. When something goes wrong it is
obvious which layer owns it.

Voice bolts on at stage 7 by replacing `input()` with the transcriber and
`print()` with Piper. Nothing else changes.
"""

from __future__ import annotations

from voice_order.agent.brain import Brain
from voice_order.agent.state import OrderSession, State


class OrderAgent:
    """A single call."""

    def __init__(self, client=None, retriever=None, persist: bool = True) -> None:
        from voice_order.db import repository
        from voice_order.llm.client import from_config

        self.session = OrderSession()
        self.session._retriever = retriever or self._default_retriever()
        self.brain = Brain(client or from_config(), self.session)
        self.persist = persist

        if persist:
            self.session.call_id = repository.open_call()

    @staticmethod
    def _default_retriever():
        from voice_order.retrieval.fusion import Retriever

        # lexical + part-number: BM25 for names, the exact/fuzzy identifier
        # index for part numbers. Measured the largest win in the project.
        # Dense is deliberately excluded -- it hurt (see docs/LEARNINGS.md).
        return Retriever.load(retrievers="lexical,part_number")

    def greeting(self) -> str:
        self.session.state = State.LISTENING
        return self.brain.greeting()

    def handle(self, utterance: str) -> str:
        """One caller utterance in, one agent reply out. Traced."""
        reply = self.brain.say(utterance)

        if self.persist:
            from voice_order.db import repository

            repository.append_turn(
                self.session.call_id,
                {
                    "said": utterance,
                    "reply": reply,
                    "tools": self.brain.tool_log[-1] if self.brain.tool_log else {},
                    "cart": self.session.read_cart(),
                },
            )
            repository.save_cart(self.session.call_id, self.session.read_cart()["lines"])
        return reply

    def close(self) -> list[str]:
        if self.persist:
            from voice_order.db import repository

            repository.close_call(self.session.call_id)
        return self.session.placed_order_ids

    # ------------------------------------------------------------ drivers --

    def run_text(self) -> None:
        """Type at it. The stage 6 way to use this."""
        print(f"agent: {self.greeting()}")
        try:
            while self.session.state is not State.CLOSED:
                said = input("you:   ").strip()
                if not said:
                    continue
                if said.lower() in {"quit", "exit", "hangup"}:
                    break
                print(f"agent: {self.handle(said)}")
        except (EOFError, KeyboardInterrupt):
            print()
        finally:
            orders = self.close()
            if orders:
                print(f"\n[{len(orders)} order line(s) written, call {self.session.call_id[:8]}]")
            else:
                print(f"\n[no order placed, call {self.session.call_id[:8]}]")

    def run_live(self) -> None:
        """Microphone in, speaker out. Stage 7."""
        raise NotImplementedError("stage 7")
