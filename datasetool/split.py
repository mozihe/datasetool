from __future__ import annotations

import os
import shutil
import random
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Optional, Set

from .exceptions import MismatchError

logger = logging.getLogger("datasetool")
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = "[%(levelname)s] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


DEFAULT_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]


@dataclass
class SplitConfig:
    images_dir: Path
    labels_dir: Path
    output_dir: Path
    val_ratio: float = 0.2
    test_ratio: float = 0.0
    seed: Optional[int] = None
    image_extensions: Iterable[str] = field(default_factory=lambda: DEFAULT_EXTS)
    strict: bool = True
    copy_mode: str = "copy"           # copy | move | symlink | hardlink
    dry_run: bool = False
    keep_structure: bool = False
    use_progress: bool = True


def _ensure_split_ratios(val_ratio: float, test_ratio: float) -> None:
    if not (0.0 <= val_ratio < 1.0) or not (0.0 <= test_ratio < 1.0):
        raise ValueError("val_ratio 与 test_ratio 必须在 [0, 1) 区间。")
    if (val_ratio + test_ratio) >= 1.0:
        raise ValueError("val_ratio + test_ratio 必须小于 1。")


def _list_images(images_dir: Path, exts: Iterable[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    exts_lower = {e.lower() for e in exts}
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts_lower:
            out[p.stem] = p
    return out


def _list_labels(labels_dir: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in labels_dir.rglob("*.txt"):
        if p.is_file():
            out[p.stem] = p
    return out


def _mkdirs(base: Path, need_test: bool) -> Dict[str, Path]:
    layout = {
        "images/train": base / "images" / "train",
        "images/val": base / "images" / "val",
        "labels/train": base / "labels" / "train",
        "labels/val": base / "labels" / "val",
    }
    if need_test:
        layout["images/test"] = base / "images" / "test"
        layout["labels/test"] = base / "labels" / "test"

    for p in layout.values():
        p.mkdir(parents=True, exist_ok=True)
    return layout


def _transfer(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    elif mode == "symlink":
        if dst.exists():
            dst.unlink()
        os.symlink(os.path.abspath(src), dst)
    elif mode == "hardlink":
        if dst.exists():
            dst.unlink()
        os.link(src, dst)
    else:
        raise ValueError(f"未知 copy_mode: {mode}")


def _maybe_progress(it, use_progress: bool, total: Optional[int] = None):
    if not use_progress:
        return it
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm(it, total=total)
    except Exception:
        return it


def split_dataset(cfg: SplitConfig) -> Tuple[List[str], List[str], List[str]]:
    """
    执行划分，返回 (train_ids, val_ids, test_ids[可能为空])
    """
    _ensure_split_ratios(cfg.val_ratio, cfg.test_ratio)

    if cfg.seed is not None:
        random.seed(cfg.seed)

    img_map = _list_images(cfg.images_dir, cfg.image_extensions)
    lbl_map = _list_labels(cfg.labels_dir)

    img_ids: Set[str] = set(img_map.keys())
    lbl_ids: Set[str] = set(lbl_map.keys())

    both = sorted(img_ids & lbl_ids)
    miss_img = sorted(lbl_ids - img_ids)
    miss_lbl = sorted(img_ids - lbl_ids)

    if miss_img or miss_lbl:
        msg = f"数据集不匹配: 缺少图片={len(miss_img)}，缺少标签={len(miss_lbl)}"
        if cfg.strict:
            raise MismatchError(msg + f"\n缺图示例: {miss_img[:5]} 缺标注示例: {miss_lbl[:5]}")
        else:
            logger.warning(msg + "（strict=False，自动跳过不匹配样本）")

    ids = list(both)
    random.shuffle(ids)

    n = len(ids)
    n_val = int(n * cfg.val_ratio)
    n_test = int(n * cfg.test_ratio)

    val_ids = ids[:n_val]
    test_ids = ids[n_val:n_val + n_test]
    train_ids = ids[n_val + n_test:]

    layout = _mkdirs(cfg.output_dir, need_test=(n_test > 0))

    def rel_keep(p: Path, root: Path) -> Path:
        if not cfg.keep_structure:
            return Path(p.name)
        return p.relative_to(root)

    for split_name, split_ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        if not split_ids:
            continue
        it = _maybe_progress(split_ids, cfg.use_progress, total=len(split_ids))
        for sid in it:
            img_src = img_map[sid]
            lbl_src = lbl_map[sid]

            img_rel = rel_keep(img_src, cfg.images_dir)
            lbl_rel = rel_keep(lbl_src, cfg.labels_dir)

            img_dst = layout[f"images/{split_name}"] / img_rel
            lbl_dst = layout[f"labels/{split_name}"] / lbl_rel

            if cfg.dry_run:
                logger.info(f"[DRY] {cfg.copy_mode} {img_src} -> {img_dst}")
                logger.info(f"[DRY] {cfg.copy_mode} {lbl_src} -> {lbl_dst}")
                continue

            _transfer(img_src, img_dst, cfg.copy_mode)
            _transfer(lbl_src, lbl_dst, cfg.copy_mode)

    logger.info(f"划分完成: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} "
                f"(total={len(both)}, skipped={len(miss_img)+len(miss_lbl)})")
    return train_ids, val_ids, test_ids
