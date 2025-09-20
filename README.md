# datasetool

[![PyPI version](https://badge.fury.io/py/datasetool.svg)](https://badge.fury.io/py/datasetool)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个轻巧而强大的 YOLO 风格数据集划分工具。

---

## ✨ 功能特性

- **多种划分比例**: 支持按比例划分训练集、验证集和测试集。
- **高可复现性**:可通过设置随机种子确保每次划分结果一致。
- **灵活的文件处理**: 支持四种文件转移模式：`copy`（复制）、`move`（移动）、`symlink`（符号链接）、`hardlink`（硬链接）。
- **严格模式**: 可选的严格模式，确保图片和标签文件一一对应，否则中断操作。
- **保留目录结构**: 可选择在输出目录中保留原始的子目录结构。
- **干预（Dry Run）**: 支持“只打印不执行”模式，方便预览划分结果。
- **自定义图片格式**: 支持自定义需要处理的图片文件扩展名。
- **进度条显示**: 在处理大量文件时显示进度条，方便跟踪进度（依赖 `tqdm`）。

## 📦 安装

1.  通过 `pip` 从 PyPI 安装 (推荐):

    ```shell
    pip install datasetool
    ```

2.  或者从源码本地安装:

    ```shell
    git clone https://github.com/mozihe/datasetool.git
    cd datasetool
    pip install .
    ```

## 🚀 使用方法

### 命令行接口 (CLI)

本工具的核心命令是 `dtsplit`。

#### 基本用法

假设您的数据集结构如下:

```
my_dataset/
├── images/
│   ├── 1.jpg
│   ├── 2.png
│   └── ...
└── labels/
    ├── 1.txt
    ├── 2.txt
    └── ...
```

您可以执行以下命令，将数据集划分为 80% 训练集、10% 验证集和 10% 测试集：

```shell
dtsplit --images ./my_dataset/images --labels ./my_dataset/labels --out ./my_dataset_split --val 0.1 --test 0.1
```

执行后，会在 `my_dataset_split` 目录下生成如下结构：

```
my_dataset_split/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

#### 参数详解

```
usage: dtsplit [-h] --images IMAGES --labels LABELS --out OUT [--val VAL] [--test TEST] [--seed SEED]
               [--exts EXTS] [--strict] [--copy-mode {copy,move,symlink,hardlink}] [--dry-run]
               [--keep-structure] [--no-progress] [-V]

Split YOLO-style dataset into train/val(/test).

options:
  -h, --help            show this help message and exit
  --images IMAGES       图片根目录 (必需)
  --labels LABELS       标签根目录 (.txt) (必需)
  --out OUT             输出目录 (必需)
  --val VAL             验证集比例，默认 0.2
  --test TEST           测试集比例，默认 0
  --seed SEED           随机种子（可复现）
  --exts EXTS           图片扩展名，逗号分隔 (默认: .jpg,.jpeg,.png,.bmp,.tiff,.tif)
  --strict              开启严格模式：一旦图片和标签不匹配直接报错
  --copy-mode {copy,move,symlink,hardlink}
                        文件转移方式（默认 copy）
  --dry-run             只打印文件转移计划，不实际执行
  --keep-structure      在输出目录中保留原 images/labels 的子目录结构
  --no-progress         不显示进度条
  -V, --version         显示版本号
```

### 作为 Python 库使用 (API)

您也可以在自己的 Python 脚本中导入 `datasetool` 并使用其核心功能。

```python
from pathlib import Path
from datasetool.split import SplitConfig, split_dataset

# 1. 配置划分参数
cfg = SplitConfig(
    images_dir=Path("./my_dataset/images"),
    labels_dir=Path("./my_dataset/labels"),
    output_dir=Path("./my_dataset_split"),
    val_ratio=0.2,
    test_ratio=0.1,
    seed=42,          # 为了可复现
    copy_mode="copy", # 使用复制模式
    strict=True,      # 开启严格模式
)

# 2. 执行划分
# 函数会返回一个元组，包含三个列表：train_ids, val_ids, test_ids
train_ids, val_ids, test_ids = split_dataset(cfg)

print(f"划分完成!")
print(f"训练集样本数: {len(train_ids)}")
print(f"验证集样本数: {len(val_ids)}")
print(f"测试集样本数: {len(test_ids)}")

```

## 📜 许可证

本项目基于 [MIT License](https://opensource.org/licenses/MIT)。