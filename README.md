# EPIC-KITCHENS データセット処理ツール

EPIC-KITCHENS データセットと VISOR アノテーションを処理し、インストラクションベースの RGB フレームとセグメンテーションマスクを抽出します。

## 概要

2 段階のパイプライン:

1. `0_split_epic_kitchencs.py`: EPIC-KITCHENS からインストラクションベースの画像を抽出
2. `1_parse_visor_dataset.py`: VISOR マスクをインストラクションフレームと統合

**対象スプリット**: 現在は train スプリットのみ。コード内のパス(`annotation_path`, `visor_annotations_dir`)を書き換えることで validation/test スプリットにも対応可能。

## セットアップ

```bash
uv sync
```

## 必要なデータセット

- **EPIC-KITCHENS**: https://github.com/epic-kitchens/epic-kitchens-download-scripts
- **VISOR**: https://data.bris.ac.uk/data/dataset/2v6cgv1x04ol22qp9rm9x2j6a7 ([詳細](https://epic-kitchens.github.io/VISOR/))
- **アノテーション CSV**: https://github.com/epic-kitchens/epic-kitchens-100-annotations

### データ配置

ダウンロードしたデータセットを以下のように配置してください:

```
/mnt/disk3/EpicKitchen/
├── EPIC-KITCHENS/
│   └── {participant_id}/
│       └── rgb_frames/
│           └── {video_id}/
│               ├── frame_0000000000.jpg
│               ├── frame_0000000001.jpg
│               └── ...
├── visor_data/
│   └── GroundTruth-SparseAnnotations/
│       └── annotations/
│           └── train/
│               ├── P01_01.json
│               ├── P01_03.json
│               └── ...
├── epic-kitchens-100-annotations/
│   ├── EPIC_100_train.csv
│   ├── EPIC_100_validation.csv
│   └── EPIC_100_test.csv
└── data/  # 出力先(自動生成)
```

## 使用方法

### ステップ 1: フレーム抽出

```bash
python 0_split_epic_kitchencs.py
```

- `EPIC_100_train.csv`から各インストラクションのフレームを抽出
- 出力: `data/images/{instruction_id}/`, `data/epic_kitchens_100_train.json`

### ステップ 2: マスク統合

```bash
python 1_parse_visor_dataset.py
```

- VISOR アノテーションとマッチングしてマスクを生成
- デフォルト: `epic_annotation["noun"]`のオブジェクトを抽出(1_parse_visor_dataset.py:59)
- 出力: `data/sparse_masks/{instruction_id}/`

**カスタマイズ例**:

```python
# 1_parse_visor_dataset.py:59 を変更
target_object = "left hand"  # 左手のマスクを抽出
```

### ステップ 3: フレームレートのダウンサンプリング (オプション)

```bash
python 2_down_sampling_fps.py
```

- 60 FPS の画像・マスクシーケンスを 15 FPS にダウンサンプリング
- 元データは `data/original_images/`, `data/original_masks/` に自動的にリネーム
- 出力: ダウンサンプリング後のデータが `data/images/`, `data/masks/` に保存

**デフォルト設定**:
- `original_fps`: 60
- `target_fps`: 15

**カスタマイズ**:

```python
# 2_down_sampling_fps.py:42-43 を変更
original_fps = 60
target_fps = 10  # 例: 10 FPS にダウンサンプリング
```

## 出力フォーマット

### JSON 構造

```json
{
  "id": 0,
  "video_id": "P01_01",
  "instruction": "open door",
  "noun": "door",
  "image_dir": "data/images/0",
  "sparse_mask_path": ["data/sparse_masks/0/0.png", ...]
}
```

### マスク

- グレースケール PNG (1920x1080)
- 背景: 0, オブジェクト: 255

## 注意事項

- **FPS**: EPIC-KITCHENS のビデオは 60 FPS で記録されています
- VISOR はスパースアノテーション(全フレームにはない)
- `sparse_mask_path`は`image_dir`より少ない場合がある

## ライセンス

[EPIC-KITCHENS](https://epic-kitchens.github.io/) と [VISOR](https://epic-kitchens.github.io/VISOR/) のライセンスに従ってください。
