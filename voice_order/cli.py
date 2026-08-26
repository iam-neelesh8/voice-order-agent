"""One entry point per stage. `python -m voice_order --help`.

Commands map onto the stages in docs/ARCHITECTURE.md, so the build order is
visible from the CLI itself.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="voice-order", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    # stage 1
    db = sub.add_parser("db", help="stage 1 - schema and health")
    db.add_argument("action", choices=["init", "health"])

    cat = sub.add_parser("catalog", help="stage 1 - build the product catalog")
    cat.add_argument("--force", action="store_true")

    # stage 2
    idx = sub.add_parser("index", help="stage 2 - build retrieval indexes")
    idx.add_argument("which", nargs="?", default="all",
                     choices=["lexical", "dense", "part-number", "all"])

    q = sub.add_parser("query", help="stage 2 - retrieve for a typed query")
    q.add_argument("text")
    q.add_argument("--top-k", type=int, default=10)

    # stage 3
    gen = sub.add_parser("gen-queries", help="stage 3 - generate order queries")
    gen.add_argument("--n", type=int, default=2000)
    gen.add_argument("--split", choices=["dev", "test"], default="dev")

    # stage 4
    aud = sub.add_parser("gen-audio", help="stage 4 - synthesize and degrade the spoken set")
    aud.add_argument("--split", choices=["dev", "test"], default="dev")

    # stages 2-8
    ev = sub.add_parser("eval", help="run the evaluation for a stage")
    ev.add_argument("target", choices=["typed", "spoken", "end-to-end", "human"])
    ev.add_argument("--split", choices=["dev", "test"], default="dev")
    ev.add_argument("--condition", default="phone")
    # stage 5 ablation -- each retrieval component switched on separately
    ev.add_argument("--nbest", action="store_true")
    ev.add_argument("--part-number", action="store_true")

    # stages 6-7
    call = sub.add_parser("call", help="stage 6/7 - run the agent")
    call.add_argument("--audio", help="recorded call; omit for live mic (stage 7)")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raise NotImplementedError(f"{args.command}: not built yet, see docs/ARCHITECTURE.md")


if __name__ == "__main__":
    raise SystemExit(main())
