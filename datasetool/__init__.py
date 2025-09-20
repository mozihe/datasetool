"""datasetool: A tiny yet robust dataset split tool for YOLO-style datasets."""
from __future__ import annotations
from datasetool.split import SplitConfig, split_dataset

__all__ = ["__version__", "SplitConfig", "split_dataset"]
__version__ = "0.1.2"

