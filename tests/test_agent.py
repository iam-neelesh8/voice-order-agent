"""Stage 6 -- the conversation, tested without a model running.

Every test here scripts the model's replies. That is the point of the hybrid
split: the cart, the thresholds, the tool validation and the ordering logic
are all outside the model, so all of them can be verified deterministically in
milliseconds instead of by talking to a 7B model and hoping.

What is *not* tested here is whether a real model chooses sensible tool calls.
That needs a real model and belongs in a separate, slower suite.
"""

from __future__ import annotations

import pytest

from voice_order.agent.brain import Brain, scripted_reply
from voice_order.agent.state import OrderSession
from voice_order.llm import tools as tool_defs
from voice_order.llm.client import FakeClient
from voice_order.types import Candidate, Product


def product(asin="B001", title="AC Delco 41-993 Professional Spark Plug", price=8.99):
    return Product(
        parent_asin=asin,
        title=title,
        category="Automotive",
        store="ACDelco",
        price=price,
        part_numbers=["41993"],
    )


class StubRetriever:
    """Returns whatever the test says the catalog found."""

    def __init__(self, candidates: list[Candidate]) -> None:
        self.candidates = candidates
        self.queries: list[str] = []

    def search_text(self, query, top_k=20, category=None, hydrate=False):
        self.queries.append(query)
        return self.candidates[:top_k]


def candidate(prod, score):
    return Candidate(parent_asin=prod.parent_asin, score=score, product=prod)


@pytest.fixture
def session():
    s = OrderSession()
    s._retriever = StubRetriever([candidate(product(), 24.0), candidate(product("B002"), 3.0)])
    return s


# ------------------------------------------------------------------- cart --


def test_the_model_cannot_add_a_product_that_does_not_exist(session):
    """The single most important guarantee here.

    A model that invents a plausible-looking id must not be able to put it on
    someone's order. Checked in code, not asked of the prompt.
    """
    result = tool_defs.execute("add_to_cart", {"product_id": "B0-INVENTED"}, session)

    assert "error" in result
    assert session.read_cart()["lines"] == []


def test_adding_requires_a_search_first(session):
    session.search("AC Delco 41-993")
    result = tool_defs.execute("add_to_cart", {"product_id": "B001", "quantity": 2}, session)

    assert result["ok"] is True
    assert result["item_count"] == 2


def test_the_total_is_arithmetic_not_generation(session):
    session.search("spark plug")
    tool_defs.execute("add_to_cart", {"product_id": "B001", "quantity": 3}, session)

    cart = session.read_cart()
    assert cart["total"] == pytest.approx(26.97)  # 3 x 8.99


def test_a_price_the_catalog_does_not_have_is_excluded_and_flagged():
    """Silently dropping it would understate the total the caller agrees to."""
    s = OrderSession()
    s._retriever = StubRetriever([candidate(product(price=None), 24.0)])
    s.search("spark plug")
    tool_defs.execute("add_to_cart", {"product_id": "B001"}, s)

    cart = s.read_cart()
    assert cart["total"] == 0.0
    assert "no price" in cart["note"]


@pytest.mark.parametrize("bad", [0, -1, 100, "many", None, True])
def test_impossible_quantities_are_refused(session, bad):
    session.search("spark plug")
    result = tool_defs.execute(
        "add_to_cart", {"product_id": "B001", "quantity": bad}, session
    )
    assert "error" in result


@pytest.mark.parametrize("given,expected", [("2", 2), (2, 2), (2.0, 2)])
def test_quantities_arrive_in_several_shapes(session, given, expected):
    """Models send strings, ints and floats for the same thing."""
    session.search("spark plug")
    result = tool_defs.execute(
        "add_to_cart", {"product_id": "B001", "quantity": given}, session
    )
    assert result["item_count"] == expected


def test_changing_your_mind_works(session):
    session.search("spark plug")
    tool_defs.execute("add_to_cart", {"product_id": "B001", "quantity": 2}, session)

    tool_defs.execute("change_quantity", {"line_number": 1, "quantity": 3}, session)
    assert session.read_cart()["item_count"] == 3

    tool_defs.execute("remove_from_cart", {"line_number": 1}, session)
    assert session.read_cart()["lines"] == []


def test_acting_on_a_line_that_is_not_there_is_an_error_not_a_crash(session):
    result = tool_defs.execute("remove_from_cart", {"line_number": 7}, session)
    assert "error" in result and "read_cart" in result["error"]


def test_an_unknown_tool_is_reported_not_raised(session):
    result = tool_defs.execute("refund_everything", {}, session)
    assert "no such tool" in result["error"]


# --------------------------------------------------------------- guidance --


def test_a_clear_winner_is_reported_as_a_strong_match(session):
    """24.0 against 3.0 is not a close call."""
    assert "Add it" in session.search("AC Delco 41-993")["guidance"]


def test_two_equally_good_matches_ask_rather_than_guess():
    """The case that produces confidently wrong orders."""
    s = OrderSession()
    s._retriever = StubRetriever(
        [candidate(product("B001"), 20.0), candidate(product("B002"), 19.8)]
    )
    guidance = s.search("spark plug")["guidance"]
    assert "Ask" in guidance or "Read the product name back" in guidance


def test_nothing_found_says_so():
    s = OrderSession()
    s._retriever = StubRetriever([])
    result = s.search("a thing we do not stock")
    assert result["matches"] == []
    assert "part number" in result["note"]


# ------------------------------------------------------------------ brain --


def test_a_whole_order_runs_end_to_end(session):
    """Search, add, read back, place -- with the model fully scripted."""
    client = FakeClient(
        [
            scripted_reply(calls=[("search_products", {"query": "AC Delco 41-993"})]),
            scripted_reply(
                calls=[("add_to_cart", {"product_id": "B001", "quantity": 2})]
            ),
            scripted_reply(text="That's two AC Delco 41-993 spark plugs. Anything else?"),
        ]
    )
    brain = Brain(client, session)

    reply = brain.say("I need two AC Delco 41-993 spark plugs")

    assert "Anything else" in reply
    assert session.read_cart()["item_count"] == 2
    assert client.tools_offered is not None


def test_tool_results_are_handed_back_to_the_model(session):
    """Otherwise it is answering blind about what it just did."""
    client = FakeClient(
        [
            scripted_reply(calls=[("search_products", {"query": "spark plug"})]),
            scripted_reply(text="I found one."),
        ]
    )
    Brain(client, session).say("spark plug please")

    last = client.seen[-1]
    tool_messages = [m for m in last if m.get("role") == "tool"]
    assert tool_messages and "B001" in tool_messages[0]["content"]


def test_a_looping_model_is_cut_off(session):
    """A model stuck on a failing tool would hold the line open forever."""
    client = FakeClient(
        [scripted_reply(calls=[("read_cart", {})]) for _ in range(20)]
        + [scripted_reply(text="Sorry, what was that?")]
    )
    reply = Brain(client, session).say("hello?")
    assert reply


def test_a_silent_model_still_says_something(session):
    """Silence on a phone call is worse than a clumsy sentence."""
    client = FakeClient([scripted_reply(text=""), scripted_reply(text="")])
    assert Brain(client, session).say("hello?").strip()


def test_the_greeting_is_fixed_not_generated(session):
    """Nothing for a model to decide, and one less thing to go wrong."""
    brain = Brain(FakeClient([]), session)
    assert brain.greeting() == brain.greeting()


def test_every_decision_branch_is_reachable():
    """A threshold that nothing can cross is dead code that looks like safety.

    `commit` was exactly that: with the original strength scale, an
    unambiguous 24-vs-3 match scored 0.795 against a 0.85 threshold, so the
    agent would have read back every single item and the confident path would
    never have run.
    """
    from voice_order.agent import policy

    def decide(*scores):
        cands = [candidate(product(f"B{i}"), s) for i, s in enumerate(scores)]
        return policy.decide(cands)

    assert decide(24.0, 3.0) == "commit"       # clear winner
    assert decide(10.0, 5.0) == "confirm"      # plausible, not certain
    assert decide(20.0, 19.8) == "clarify"     # two equally good
    assert decide(1.0) == "reask"              # nothing convincing
    assert decide() == "reask"                 # nothing at all


def test_a_thin_margin_blocks_commit_however_high_the_score():
    """Two plausible part numbers is the case that produces wrong orders."""
    from voice_order.agent import policy

    huge_but_tied = [candidate(product("B1"), 90.0), candidate(product("B2"), 89.5)]
    assert policy.decide(huge_but_tied) != "commit"


def test_readback_does_not_recite_the_catalog_title():
    """Amazon titles run to 200 characters of keywords and are unspeakable."""
    from voice_order.agent import policy

    long_title = product(title="ACDelco 41-993 Professional Iridium Spark Plug " * 6)
    spoken = policy.readback(candidate(long_title, 20.0), quantity=2)

    assert len(spoken) < 60
    assert spoken.startswith("2 ")


# ------------------------------------------------- placing the order safely --


def test_an_order_cannot_be_placed_before_the_total_is_read(session):
    """The one mistake here that costs somebody money.

    A model that reads "yes, go ahead" as agreement to a total it never said
    would otherwise place an order at a price the caller never heard. The
    prompt asks; this makes it so.
    """
    session.search("spark plug")
    tool_defs.execute("add_to_cart", {"product_id": "B001", "quantity": 2}, session)

    result = tool_defs.execute("place_order", {}, session)

    assert "error" in result
    assert "read_cart" in result["error"]
    assert session.placed_order_ids == []


def test_reading_the_cart_unlocks_placing_it(session, monkeypatch):
    recorded = []
    monkeypatch.setattr(
        "voice_order.db.repository.commit_order",
        lambda **kw: recorded.append(kw) or "order-1",
    )

    session.search("spark plug")
    tool_defs.execute("add_to_cart", {"product_id": "B001", "quantity": 2}, session)
    tool_defs.execute("read_cart", {}, session)

    result = tool_defs.execute("place_order", {}, session)

    assert result["ok"] is True
    assert len(recorded) == 1
    assert recorded[0]["quantity"] == 2


def test_changing_the_cart_after_reading_it_locks_it_again(session):
    """The caller agreed to a total for a different basket."""
    session.search("spark plug")
    tool_defs.execute("add_to_cart", {"product_id": "B001", "quantity": 1}, session)
    tool_defs.execute("read_cart", {}, session)

    tool_defs.execute("change_quantity", {"line_number": 1, "quantity": 5}, session)

    assert "error" in tool_defs.execute("place_order", {}, session)


def test_adding_returns_the_cart_without_counting_as_reading_it(session):
    """add_to_cart echoes the cart back. That is not the caller hearing it.

    Regression: the reset was written one line above a `read_cart()` call that
    promptly set the flag back to True, so the guard never fired.
    """
    session.search("spark plug")
    result = tool_defs.execute("add_to_cart", {"product_id": "B001"}, session)

    assert "total" in result                       # the model still sees it
    assert session._total_read_since_change is False


def test_an_order_can_be_placed_from_a_session_nobody_registered(tmp_path, monkeypatch):
    """Regression: orders.call_id is a foreign key.

    A session built directly -- by `check-model`, or by anything embedding the
    agent -- has a call_id with no row behind it, and place_order failed with a
    bare IntegrityError instead of working or explaining itself.
    """
    monkeypatch.setenv("VOICE_ORDER_DATA_DIR", str(tmp_path))
    from voice_order.db import repository
    from voice_order.db import session as db

    db.init_schema()
    repository.upsert_products(
        [
            Product(
                parent_asin="B001",
                title="ACDelco 41-993 Spark Plug",
                category="Automotive",
                store="ACDelco",
                price=8.99,
            )
        ]
    )

    s = OrderSession()                      # never went through open_call
    s._retriever = StubRetriever([candidate(product(), 24.0)])
    s.search("spark plug")
    tool_defs.execute("add_to_cart", {"product_id": "B001", "quantity": 2}, s)
    tool_defs.execute("read_cart", {}, s)

    result = tool_defs.execute("place_order", {}, s)

    assert result["ok"] is True
    assert repository.orders_for_call(s.call_id)[0]["quantity"] == 2
