"""The GPU round trip: export texts, embed elsewhere, import the result.

The import is the dangerous part. Both failure modes it guards against are
silent -- a wrong index does not crash, it just quietly returns the wrong
products -- and both are easy to hit in practice: reloading the catalog after
exporting, or a remote box that installed a different model variant.

So the guards get tested by actually tripping them, because a check that never
rejects anything is worse than no check at all.
"""

from __future__ import annotations

import gzip
import json
import shutil

import numpy as np
import pytest

fastembed = pytest.importorskip("fastembed", reason="needs the retrieval extra")

from voice_order.retrieval import portable  # noqa: E402
from voice_order.types import Product  # noqa: E402


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    """A tiny throwaway catalog, isolated from the real database."""
    monkeypatch.setenv("VOICE_ORDER_DB", str(tmp_path / "t.db"))
    monkeypatch.setenv("VOICE_ORDER_INDEX_DIR", str(tmp_path / "index"))
    # Without this the fixture writes its 24-product export over the real one.
    monkeypatch.setenv("VOICE_ORDER_EXPORT_DIR", str(tmp_path / "exports"))

    from voice_order.db import repository, session

    session.init_schema()
    repository.upsert_products(
        [
            Product(
                parent_asin=f"B{i:04d}",
                title=f"Bosch {1000 + i}-A Widget Model {i}",
                category="Automotive",
                store="Bosch",
                part_numbers=[f"{1000 + i}A"],
            )
            for i in range(24)
        ]
    )
    return tmp_path


@pytest.fixture
def remote_output(catalog, tmp_path):
    """What a correct GPU run hands back."""
    from voice_order.retrieval.dense import embed_texts

    path, _, _ = portable.export_embed_input()
    ids, texts = [], []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("_meta"):
                continue
            ids.append(row["id"])
            texts.append(row["text"])

    out = tmp_path / "remote"
    out.mkdir()
    np.save(out / "embeddings.npy", embed_texts(texts))
    (out / "ids.json").write_text(json.dumps({"ids": ids}), encoding="utf-8")
    return out


def test_the_export_carries_every_product(catalog):
    path, count, fingerprint = portable.export_embed_input()
    assert count == 24
    assert path.is_file()
    assert len(fingerprint) == 16


def test_a_correct_round_trip_installs_a_working_index(remote_output):
    from voice_order.retrieval.dense import DenseIndex

    report = portable.import_embeddings(remote_output)

    assert report["vectors"] == 24
    # Same model both ends means the vectors are identical, not merely close.
    assert report["verify_min_cosine"] > 0.999

    hits = DenseIndex.load().search("Bosch 1007-A Widget", top_k=3)
    assert hits[0].parent_asin == "B0007"


def test_categories_are_filled_in_locally(remote_output):
    """They are never shipped to the GPU box, so they must survive the import."""
    from voice_order.retrieval.dense import DenseIndex

    portable.import_embeddings(remote_output)
    index = DenseIndex.load()
    assert set(index.categories.tolist()) == {"Automotive"}


def test_ids_that_do_not_match_the_catalog_are_rejected(remote_output, tmp_path):
    """The catalog changed after the export -- vectors now map to wrong rows."""
    bad = tmp_path / "bad_ids"
    shutil.copytree(remote_output, bad)
    ids = json.loads((bad / "ids.json").read_text(encoding="utf-8"))["ids"]
    (bad / "ids.json").write_text(json.dumps({"ids": list(reversed(ids))}), encoding="utf-8")

    with pytest.raises(ValueError, match="diverge from the catalog"):
        portable.import_embeddings(bad)


def test_vectors_from_a_different_model_are_rejected(remote_output, tmp_path):
    """Document and query vectors must come from the same model.

    If they do not, nothing crashes -- every cosine score is just quietly
    miscalibrated, which is far worse.
    """
    bad = tmp_path / "bad_vectors"
    shutil.copytree(remote_output, bad)
    rng = np.random.default_rng(0)
    np.save(bad / "embeddings.npy", rng.random((24, 384)).astype(np.float32))

    with pytest.raises(ValueError, match="do not match locally computed"):
        portable.import_embeddings(bad)


def test_a_count_mismatch_is_rejected(remote_output, tmp_path):
    bad = tmp_path / "bad_count"
    shutil.copytree(remote_output, bad)
    np.save(bad / "embeddings.npy", np.load(remote_output / "embeddings.npy")[:10])

    with pytest.raises(ValueError, match="do not belong together"):
        portable.import_embeddings(bad)


def test_skip_verify_bypasses_only_the_model_check(remote_output, tmp_path):
    """The id check is structural and must hold even when verification is off."""
    bad = tmp_path / "skip"
    shutil.copytree(remote_output, bad)
    rng = np.random.default_rng(1)
    np.save(bad / "embeddings.npy", rng.random((24, 384)).astype(np.float32))

    report = portable.import_embeddings(bad, skip_verify=True)
    assert "verify_min_cosine" not in report

    ids = json.loads((bad / "ids.json").read_text(encoding="utf-8"))["ids"]
    (bad / "ids.json").write_text(json.dumps({"ids": list(reversed(ids))}), encoding="utf-8")
    with pytest.raises(ValueError):
        portable.import_embeddings(bad, skip_verify=True)


def test_the_export_never_escapes_its_configured_directory(catalog, tmp_path, monkeypatch):
    """Regression: the test fixture once overwrote the real 6 MB export.

    VOICE_ORDER_DB and VOICE_ORDER_INDEX_DIR were redirectable but the export
    directory was not, so a 24-product fixture silently replaced a 100k-product
    artifact. Nothing raised; the file was just wrong.
    """
    sandbox = tmp_path / "elsewhere"
    monkeypatch.setenv("VOICE_ORDER_EXPORT_DIR", str(sandbox))

    path, count, _ = portable.export_embed_input()

    assert sandbox in path.parents
    assert count == 24
