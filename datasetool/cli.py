from __future__ import annotations

import argparse
from pathlib import Path

from .split import SplitConfig, split_dataset
from . import __version__


def positive_float(x: str) -> float:
    v = float(x)
    if v < 0:
        raise argparse.ArgumentTypeError("ratio 必须 >= 0")
    return v


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dtsplit",
        description="Split YOLO-style dataset into train/val(/test)."
    )
    parser.add_argument("--images", required=True, type=Path, help="图片根目录")
    parser.add_argument("--labels", required=True, type=Path, help="标签根目录（.txt）")
    parser.add_argument("--out", required=True, type=Path, help="输出目录")
    parser.add_argument("--val", type=positive_float, default=0.2, help="验证集比例，默认 0.2")
    parser.add_argument("--test", type=positive_float, default=0.0, help="测试集比例，默认 0")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（可复现）")
    parser.add_argument("--exts", type=str, default=".jpg,.jpeg,.png,.bmp,.tiff,.tif",
                        help="图片扩展名，逗号分隔")
    parser.add_argument("--strict", action="store_true", help="开启严格模式：一旦不匹配直接报错")
    parser.add_argument("--copy-mode", choices=["copy", "move", "symlink", "hardlink"],
                        default="copy", help="文件转移方式（默认 copy）")
    parser.add_argument("--dry-run", action="store_true", help="只打印不执行")
    parser.add_argument("--keep-structure", action="store_true",
                        help="保留原 images/labels 子目录结构")
    parser.add_argument("--no-progress", action="store_true", help="不显示进度条")
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")

    args = parser.parse_args()

    cfg = SplitConfig(
        images_dir=args.images,
        labels_dir=args.labels,
        output_dir=args.out,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
        image_extensions=[e.strip() for e in args.exts.split(",") if e.strip()],
        strict=args.strict,
        copy_mode=args.copy_mode,
        dry_run=args.dry_run,
        keep_structure=args.keep_structure,
        use_progress=not args.no_progress,
    )
    split_dataset(cfg)


if __name__ == "__main__":
    main()
