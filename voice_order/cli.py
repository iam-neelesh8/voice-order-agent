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
    gen.add_argument("--split", choices=["dev", "test", "both"], default="both")
    gen.add_argument("--seed", type=int, default=20260826)

    # stage 4
    aud = sub.add_parser("gen-audio", help="stage 4 - synthesize and degrade the spoken set")
    aud.add_argument("--split", choices=["dev", "test"], default="dev")

    # stages 2-8
    ev = sub.add_parser("eval", help="run the evaluation for a stage")
    ev.add_argument("target", choices=["typed", "spoken", "end-to-end", "human"])
    ev.add_argument("--query-set", choices=["lookup", "orders"], default="lookup")
    ev.add_argument("--split", choices=["dev", "test"], default="dev")
    ev.add_argument("--condition", default="phone")
    ev.add_argument("--retrievers", default="lexical",
                    help="comma separated: lexical,dense,part_number")
    ev.add_argument("--limit", type=int, help="only the first N queries (smoke runs)")
    ev.add_argument("--category", help="restrict retrieval to one category")
    # stage 5 ablation -- each retrieval component switched on separately
    ev.add_argument("--nbest", action="store_true")
    ev.add_argument("--part-number", action="store_true")

    # stages 6-7
    call = sub.add_parser("call", help="stage 6/7 - run the agent")
    call.add_argument("--audio", help="recorded call; omit for live mic (stage 7)")

    return p


def _cmd_db(args) -> int:
    from voice_order.db import session

    if args.action == "init":
        path = session.init_schema()
        print(f"schema applied to {path}")
        return 0

    health = session.healthcheck()
    print(f"database  {health['database']}")
    print(f"          {'present' if health['exists'] else 'MISSING'}"
          f"  |  {health['size_mb']} MB")
    print()
    print("tables")
    for table, count in health["tables"].items():
        shown = "not created" if count is None else f"{count:,}"
        print(f"  {table:<24}{shown:>14}")
    print()
    print("indexes (derived -- rebuild with `voice-order index all`)")
    for name, present in health["indexes"].items():
        print(f"  {name:<24}{'present' if present else 'missing':>14}")
    return 0


def _cmd_catalog(args) -> int:
    from voice_order.catalog import load

    stats = load.build_catalog(force=args.force)
    load.report(stats)
    return 0


def _cmd_index(args) -> int:
    from voice_order.retrieval.fusion import build_all_indexes

    built = build_all_indexes(args.which)
    for name, n in built.items():
        print(f"  {name:<14}{n:>10,} documents")
    print()
    print("indexes are derived data -- rebuild after any catalog change")
    return 0


def _cmd_eval(args) -> int:
    from voice_order.evaluation import run

    if args.target == "typed":
        result = run.eval_typed_retrieval(
            query_set=args.query_set,
            split=args.split,
            limit=args.limit,
            category=args.category,
            retrievers=args.retrievers,
        )
        run.report(result)
        return 0

    raise NotImplementedError(
        f"eval {args.target}: not built yet -- see docs/ARCHITECTURE.md"
    )


def _cmd_gen_queries(args) -> int:
    from voice_order.evaluation import queries

    for split in (["dev", "test"] if args.split == "both" else [args.split]):
        rows = queries.generate_order_queries(args.n, seed=args.seed, split=split)
        path = queries.order_set_path(split)
        queries.write_split(rows, path)
        kinds: dict[str, int] = {}
        for q in rows:
            kinds[q.kind] = kinds.get(q.kind, 0) + 1
        print(f"{split:<6} {len(rows):>6,} queries -> {path}")
        for kind, n in sorted(kinds.items(), key=lambda kv: -kv[1]):
            print(f"         {kind:<16}{n:>6,}")
    return 0


def _cmd_query(args) -> int:
    from voice_order.retrieval.fusion import Retriever

    retriever = Retriever.load(retrievers="lexical")
    for i, c in enumerate(retriever.search_text(args.text, top_k=args.top_k, hydrate=True), 1):
        title = c.product.title[:78] if c.product else "(missing)"
        print(f"{i:>3}. {c.score:7.3f}  {c.parent_asin}  {title}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "db":
        return _cmd_db(args)
    if args.command == "catalog":
        return _cmd_catalog(args)
    if args.command == "index":
        return _cmd_index(args)
    if args.command == "eval":
        return _cmd_eval(args)
    if args.command == "gen-queries":
        return _cmd_gen_queries(args)
    if args.command == "query":
        return _cmd_query(args)

    raise NotImplementedError(
        f"{args.command}: not built yet -- see docs/ARCHITECTURE.md for the stage it belongs to"
    )


if __name__ == "__main__":
    raise SystemExit(main())
