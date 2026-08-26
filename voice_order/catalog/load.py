"""Stage 1 -- download, normalise, write to SQLite. The stage 1 entry point."""

from __future__ import annotations

from voice_order import config
from voice_order.catalog import normalize, stream
from voice_order.db import repository, session


def build_catalog(force: bool = False) -> dict[str, dict]:
    """Run the whole stage: stream -> normalise -> upsert.

    Returns per-category stats. Stage 1 is done when the row counts match the
    limits in `configs/catalog.yaml` *and* identifier coverage separates
    Automotive from Health_and_Household -- the second half is what says the
    extractor works, and a load that satisfies only the first half is a load
    that will quietly ruin every later number.
    """
    cfg = config.load("catalog")
    session.init_schema()

    stats: dict[str, dict] = {}

    for entry in cfg["categories"]:
        name, limit = entry["name"], int(entry["limit"])

        print(f"\n{name}")
        print(f"  streaming first {limit:,} items ...", flush=True)
        path = stream.download_category(name, limit, force=force)

        skipped = 0
        with_ids = 0

        def products():
            nonlocal skipped, with_ids
            for raw in stream.iter_raw_items(path):
                product = normalize.normalize_item(raw, name)
                if product is None:
                    skipped += 1
                    continue
                if product.part_numbers:
                    with_ids += 1
                yield product

        written = repository.upsert_products(products())
        stats[name] = {
            "written": written,
            "skipped": skipped,
            "with_part_numbers": with_ids,
            "identifier_coverage": round(with_ids / written, 3) if written else 0.0,
        }
        print(
            f"  {written:,} rows"
            f"  |  {skipped:,} skipped (missing asin/title)"
            f"  |  {stats[name]['identifier_coverage']:.1%} carry an identifier"
        )

    return stats


def report(stats: dict[str, dict]) -> None:
    """Print the stage 1 done-check."""
    cfg = config.load("catalog")
    expected = {e["name"]: int(e["limit"]) for e in cfg["categories"]}

    print("\n" + "=" * 68)
    print(f"{'category':<32}{'rows':>10}{'expected':>10}{'ids':>10}")
    print("-" * 68)
    total = 0
    for name, want in expected.items():
        got = stats.get(name, {}).get("written", 0)
        cov = stats.get(name, {}).get("identifier_coverage", 0.0)
        flag = "" if got >= want else "  <-- short"
        total += got
        print(f"{name:<32}{got:>10,}{want:>10,}{cov:>9.1%}{flag}")
    print("-" * 68)
    print(f"{'total':<32}{total:>10,}{sum(expected.values()):>10,}")

    sources = repository.part_number_sources()
    if sources:
        print("\nidentifiers by source: " + "  ".join(f"{k}={v:,}" for k, v in sources.items()))

    cov = {k: v.get("identifier_coverage", 0.0) for k, v in stats.items()}
    hard = cov.get("Automotive", 0.0)
    easy = cov.get("Health_and_Household", 0.0)
    print(f"\nAutomotive {hard:.1%} vs Health_and_Household {easy:.1%}")
    if hard - easy < 0.20:
        print(
            "  ! These should be far apart. If they are not, the extractor in\n"
            "    normalize.py is wrong and every later result measures the\n"
            "    wrong thing. Fix this before starting stage 2."
        )
    else:
        print("  the category mix is doing its job -- identifiers separate the two")
