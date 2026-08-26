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

    # run the heavy embedding step elsewhere
    exp = sub.add_parser("export-embed-input",
                         help="write the catalog texts for a GPU box to embed")
    exp.add_argument("--out", help="destination .jsonl.gz")

    expa = sub.add_parser("export-asr-input",
                          help="bundle audio + manifest for a GPU box to transcribe")
    expa.add_argument("--split", choices=["dev", "test"], default="dev")
    expa.add_argument("--condition", action="append",
                      help="limit to one condition (repeatable); default is all")

    impt = sub.add_parser("import-transcripts",
                          help="install transcripts produced elsewhere, with checks")
    impt.add_argument("source", help="folder of *.jsonl, or a single file")
    impt.add_argument("--split", choices=["dev", "test"], default="dev")

    imp = sub.add_parser("import-embeddings",
                         help="install embeddings built elsewhere, with checks")
    imp.add_argument("source", help="folder holding embeddings.npy and ids.json")
    imp.add_argument("--skip-verify", action="store_true",
                     help="skip the model-match check (not recommended)")

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


def _cmd_export_asr_input(args) -> int:
    from voice_order.asr import portable

    path, stats = portable.export_asr_input(args.split, args.condition)
    print(f"{stats['clips']:,} clips -> {path}  ({stats['mb']} MB)")
    print(f"conditions: {', '.join(stats['conditions'])}")
    if stats["missing"]:
        print(f"  ! {stats['missing']:,} manifest rows had no audio file")
    print()
    print("next:")
    print("  1. open notebooks/transcribe_gpu.py on Kaggle or Colab (GPU runtime)")
    print("  2. upload this zip, run the cell, download transcripts.zip")
    print("  3. unzip it, then:  voice-order import-transcripts transcripts")
    return 0


def _cmd_import_transcripts(args) -> int:
    from voice_order.asr import portable

    installed = portable.import_transcripts(args.source, args.split)
    print(f"{'condition':<16}{'model':<14}{'clips':>8}{'coverage':>10}")
    print("-" * 48)
    for row in installed:
        print(f"{row['condition']:<16}{row['model']:<14}{row['clips']:>8,}"
              f"{row['coverage']:>9.1%}")
        if row["empty"]:
            print(f"  ! {row['empty']:,} clips produced no transcript at all")
    print()
    print("now:  voice-order eval spoken --condition <condition>")
    return 0


def _cmd_export_embed_input(args) -> int:
    from voice_order.retrieval import portable

    path, count, fingerprint = portable.export_embed_input(args.out)
    size_mb = path.stat().st_size / 1e6
    print(f"{count:,} texts -> {path}  ({size_mb:.1f} MB)")
    print(f"catalog fingerprint  {fingerprint}")
    print()
    print("next:")
    print("  1. open notebooks/embed_catalog_gpu.py on Kaggle or Colab (GPU runtime)")
    print("  2. upload this file, run the cell, download catalog_embeddings.zip")
    print("  3. unzip it, then:  voice-order import-embeddings catalog_embeddings")
    return 0


def _cmd_import_embeddings(args) -> int:
    from voice_order.retrieval import portable

    report = portable.import_embeddings(args.source, skip_verify=args.skip_verify)
    print(f"vectors        {report['vectors']:,} x {report['dim']}")
    print(f"catalog rows   {report['catalog_rows']:,}  (ids match)")
    print(f"fingerprint    {report['fingerprint']}")
    if "verify_min_cosine" in report:
        print(f"model check    min cosine {report['verify_min_cosine']:.5f}"
              f"  mean {report['verify_mean_cosine']:.5f}")
    else:
        print("model check    SKIPPED")
    print(f"installed to   {report['installed']}")
    print()
    print("dense retrieval is now available:")
    print("  voice-order eval typed --query-set orders --retrievers lexical,dense")
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
    if args.command == "transcribe":
        return _cmd_transcribe(args)
    if args.command == "gen-audio":
        return _cmd_gen_audio(args)
    if args.command == "export-asr-input":
        return _cmd_export_asr_input(args)
    if args.command == "import-transcripts":
        return _cmd_import_transcripts(args)
    if args.command == "export-embed-input":
        return _cmd_export_embed_input(args)
    if args.command == "import-embeddings":
        return _cmd_import_embeddings(args)
    if args.command == "query":
        return _cmd_query(args)

    raise NotImplementedError(
        f"{args.command}: not built yet -- see docs/ARCHITECTURE.md for the stage it belongs to"
    )


if __name__ == "__main__":
    raise SystemExit(main())
