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

    sd = sub.add_parser("seed",
                        help="stage 1 - invent the shop data Amazon does not carry")
    sd.add_argument("--seed", type=int, default=20260827)

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
    aud.add_argument("--limit", type=int, help="only the first N queries (smoke runs)")
    aud.add_argument("--force", action="store_true", help="re-render existing clips")

    tr = sub.add_parser("transcribe", help="stage 4 - run ASR over a condition")
    tr.add_argument("--split", choices=["dev", "test"], default="dev")
    tr.add_argument("--condition", default="phone")
    tr.add_argument("--model", help="override the model in configs/asr.yaml")
    tr.add_argument("--n-best", type=int,
                    help="decodes per clip; 1 is ~5x faster and enough for stage 4")
    tr.add_argument("--limit", type=int)

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
    mdl = sub.add_parser("model", help="stage 6 - show or switch the LLM (ollama/gemini)")
    mdl.add_argument("name", nargs="?", help="profile to switch to; omit to show current")

    web = sub.add_parser("serve", help="stage 6 - a local demo page you can watch")
    web.add_argument("--port", type=int, default=8000)
    web.add_argument("--host", default="127.0.0.1")

    chk = sub.add_parser("check-model",
                         help="stage 6 - can this model actually drive the agent?")
    chk.add_argument("--model", help="override configs/agent.yaml for this run")

    call = sub.add_parser("call", help="stage 6/7 - take an order")
    call.add_argument("--text", action="store_true",
                      help="type at it instead of speaking (stage 6)")
    call.add_argument("--audio", help="recorded call; omit for live mic (stage 7)")
    call.add_argument("--no-save", action="store_true",
                      help="do not write the call or order to the database")

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


def _cmd_seed(args) -> int:
    from voice_order.catalog import seed

    seed.report(seed.seed_all(args.seed))
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

    if args.target == "spoken":
        result = run.eval_spoken_retrieval(
            split=args.split,
            condition=args.condition,
            nbest=args.nbest,
            retrievers=args.retrievers,
            limit=args.limit,
        )
        run.report(result)
        return 0

    raise NotImplementedError(
        f"eval {args.target}: not built yet -- see docs/ARCHITECTURE.md"
    )


def _cmd_transcribe(args) -> int:
    from voice_order.asr import batch

    print(f"transcribing {args.split}/{args.condition} ...", flush=True)
    _, stats = batch.transcribe_manifest(
        args.split, args.condition, model=args.model,
        n_best=args.n_best, limit=args.limit,
    )
    print()
    print(f"model         {stats['model']}")
    print(f"clips         {stats['total']:,}  "
          f"(resumed {stats['resumed']:,}, new {stats['transcribed']:,})")
    if stats["missing_audio"]:
        print(f"missing audio {stats['missing_audio']:,}")
    if stats["empty"]:
        print(f"empty results {stats['empty']:,}")
    print(f"elapsed       {stats['elapsed_s']:.0f}s")
    print(f"written to    {stats['path']}")
    return 0


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


def _cmd_gen_audio(args) -> int:
    from voice_order.evaluation import audio

    stats = audio.build_spoken_set(args.split, limit=args.limit, force=args.force)
    print(f"synthesised {stats['synthesised']:,}  |  reused {stats['skipped']:,}"
          f"  |  wrote {stats['clips']:,} clips")
    print()
    print(f"{'condition':<16}{'clips':>8}{'audio':>12}{'size':>10}")
    print("-" * 46)
    total_mb = 0.0
    for condition, info in audio.describe(args.split).items():
        total_mb += info["mb"]
        mins = info["seconds"] / 60
        print(f"{condition:<16}{info['clips']:>8,}{mins:>10.1f}m{info['mb']:>9.1f}M")
    print("-" * 46)
    print(f"{'total':<16}{'':>8}{'':>12}{total_mb:>9.1f}M")
    print()
    print(f"manifest: {audio.split_dir(args.split) / audio.MANIFEST_NAME}")
    return 0


def _cmd_model(args) -> int:
    import re

    from voice_order import config
    from voice_order.llm.client import active_profile

    cfg = config.load("agent")
    profiles = cfg.get("llm.profiles", {})

    if not args.name:
        current = active_profile()
        print(f"active LLM: {current}")
        for name, spec in profiles.items():
            mark = " <-- active" if name == current else ""
            print(f"  {name:<18}{spec.get('model'):<22}{mark}")
        return 0

    if args.name not in profiles:
        print(f"unknown profile {args.name!r}. Known: {', '.join(profiles)}")
        return 1

    # Rewrite just the `active:` line, leaving the rest of the file untouched.
    path = config.CONFIG_DIR / "agent.yaml"
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"(?m)^(\s*active:\s*).*$", rf"\g<1>{args.name}", text, count=1)
    path.write_text(text, encoding="utf-8")
    config.load.cache_clear()
    print(f"switched to {args.name} ({profiles[args.name].get('model')})")
    return 0


def _cmd_serve(args) -> int:
    from voice_order.web import serve

    serve(host=args.host, port=args.port)
    return 0


def _cmd_check_model(args) -> int:
    from voice_order.agent import check

    result = check.run(model=args.model)
    check.report(result)
    return 0 if result.passed == result.total else 1


def _cmd_call(args) -> int:
    from voice_order.agent.loop import OrderAgent

    if not args.text:
        raise NotImplementedError(
            "voice calls arrive in stage 7. Use `voice-order call --text` to "
            "type at the agent -- it exercises the whole conversation."
        )

    agent = OrderAgent(persist=not args.no_save)
    agent.run_text()
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
    if args.command == "seed":
        return _cmd_seed(args)
    if args.command == "index":
        return _cmd_index(args)
    if args.command == "eval":
        return _cmd_eval(args)
    if args.command == "gen-queries":
        return _cmd_gen_queries(args)
    if args.command == "transcribe":
        return _cmd_transcribe(args)
    if args.command == "gen-audio":
        return _cmd_gen_audio(args)
    if args.command == "query":
        return _cmd_query(args)
    if args.command == "call":
        return _cmd_call(args)
    if args.command == "check-model":
        return _cmd_check_model(args)
    if args.command == "model":
        return _cmd_model(args)
    if args.command == "serve":
        return _cmd_serve(args)

    raise NotImplementedError(
        f"{args.command}: not built yet -- see docs/ARCHITECTURE.md for the stage it belongs to"
    )


if __name__ == "__main__":
    raise SystemExit(main())
