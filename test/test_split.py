from __future__ import annotations
import os
from pathlib import Path

from datasetool.split import SplitConfig, split_dataset


def _touch(p: Path, text: str = "") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def test_basic_split(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    out = tmp_path / "out"

    for i in range(10):
        _touch(images / f"{i}.jpg", "img")
        _touch(labels / f"{i}.txt", "0 0.5 0.5 0.2 0.2")

    cfg = SplitConfig(
        images_dir=images,
        labels_dir=labels,
        output_dir=out,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=42,
        strict=True,
        copy_mode="copy",
        dry_run=False,
        keep_structure=False,
        use_progress=False,
    )
    train_ids, val_ids, test_ids = split_dataset(cfg)

    assert len(train_ids) + len(val_ids) + len(test_ids) == 10
    for sid in train_ids:
        assert (out / "images/train" / f"{sid}.jpg").exists()
        assert (out / "labels/train" / f"{sid}.txt").exists()
