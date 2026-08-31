# Point_net_for_my_research

UAV（ドローン）空撮から復元した**水田の3次元点群**を入力として、**区画ごとの収量（g/m²）を回帰予測**する研究用のコード。
点群分類の定番実装 [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)（PointNet / PointNet++ の PyTorch 実装）をベースに、
**点群から収量を予測できるよう改良した派生リポジトリ**。

---

## 研究の背景と意義

### 背景

日本の稲作では、圃場の生育状況を**3次元的に**把握することが、収量予測や管理効率化にとって重要。

### 収量予測のメリット

| 観点 | 内容 |
| --- | --- |
| 経営面 | 不作のダメージを事前に見積もり、資金繰りをスムーズに行える |
| 作業面 | 収穫に向けた農機・肥料・人員の準備を最適化できる |
| 売り先の面 | 事前に買い手へ納品予定量を伝えられ、取引がしやすくなる |

### 圃場を3次元点群にするメリット

- **生育状況の可視化・定量化**：広大な圃場を見回る労力を削減しながら、生育ムラや倒伏のリスクを発見できる
- **圃場記録のデジタル化・比較**：年ごと・区画ごとの3Dデータを蓄積・比較することで圃場の変化を客観的に追跡でき、年ごとの施策の影響を精密に記録できる

### 収量予測に3次元点群を用いるメリット

- **従来の構造化データによる収量予測**：多数の実測値を人手で計測する必要があり、労力がかかる
- **ドローンRGB画像による収量予測**：RGB画像は「天気」や「影の出方」で見え方が変わるためロバストではない。マルチスペクトルカメラを使えば改善できるが非常に高価
- **3次元点群による収量予測**：ドローン画像と収量データさえあればよい。純粋な稲の構造のみから予測するためロバスト性が高く、稲の構造そのものを入力とするため出力の説明性も高い。さらに、収量予測をしながら点群による管理効率化も同時に行える

---

## ベース実装との関係

本リポジトリは [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) をベースにしている。

| 区分 | ファイル | 由来 |
| --- | --- | --- |
| 収量回帰（本リポジトリの追加・改良分） | [data_utils/RicePaddyDataLoader.py](data_utils/RicePaddyDataLoader.py), [train_regression.py](train_regression.py), [models/pointnet_regression.py](models/pointnet_regression.py), [models/pointnet2_regression.py](models/pointnet2_regression.py) | 新規 |
| 分類（ModelNet40、ベースからほぼそのまま継承） | [train_classification.py](train_classification.py), [test_classification.py](test_classification.py), [data_utils/ModelNetDataLoader.py](data_utils/ModelNetDataLoader.py), [models/pointnet_cls.py](models/pointnet_cls.py), [models/pointnet2_cls_msg.py](models/pointnet2_cls_msg.py) | 継承 |
| 共通ユーティリティ | [models/pointnet_utils.py](models/pointnet_utils.py), [models/pointnet2_utils.py](models/pointnet2_utils.py), [utils/provider.py](utils/provider.py) | 継承 |

分類部分（ModelNet40）の詳細やデータ取得手順はベースリポジトリを参照すること。以降は**収量回帰**を中心に説明する。

---

## リポジトリ構成

```
Point_net_for_my_research/
├── train_regression.py            # 収量回帰の学習スクリプト（本体）
├── train_classification.py        # 分類の学習（継承）
├── test_classification.py         # 分類の評価（継承）
├── Dockerfile                     # 実行環境（PyTorch 2.0.1 / CUDA 11.7）
├── data_utils/
│   ├── RicePaddyDataLoader.py     # 水田点群 + 収量の DataLoader（本体）
│   └── ModelNetDataLoader.py      # ModelNet 用 DataLoader（継承。pc_normalize はここを共用）
├── models/
│   ├── pointnet_regression.py     # PointNet 回帰モデル（追加）
│   ├── pointnet2_regression.py    # PointNet++ (MSG) 回帰モデル（追加）
│   ├── pointnet_cls.py            # PointNet 分類モデル（継承）
│   ├── pointnet2_cls_msg.py       # PointNet++ 分類モデル（継承）
│   ├── pointnet_utils.py          # STN3d / STNkd / PointNetEncoder など（継承）
│   └── pointnet2_utils.py         # Set Abstraction など（継承）
├── utils/
│   └── provider.py                # 点群データ拡張（継承）
├── data/                          # データ配置先（.gitignore 済み。下記参照）
└── log/                           # 学習成果物の出力先（.gitignore 済み）
```

---

## 必要環境 / セットアップ

### Docker（推奨）

[Dockerfile](Dockerfile) は `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel` をベースに、
`numpy<2` / `tqdm` / `matplotlib` / `scikit-learn` / `plyfile` を追加インストールする。

```bash
docker build -t pointnet-rice .

# data/ と log/ は .dockerignore で除外されるため、実行時にマウントする
docker run --gpus all -it --rm \
  -v "$(pwd)/data:/workspace/data" \
  -v "$(pwd)/log:/workspace/log" \
  pointnet-rice \
  python train_regression.py --data_root data
```

### 手動セットアップ

- Python 3.10 相当（ベースイメージと同等）
- PyTorch 2.0.1（CUDA 11.7 ビルド）
- `pip install "numpy<2" tqdm matplotlib scikit-learn plyfile`

> **`numpy<2` の固定について**：ベースイメージの torch 2.0.1 は NumPy 1.x 向けにビルドされているため、
> バージョン指定なしで入れると NumPy 2 系に上がり、ABI 不整合でクラッシュする。

---

## データ準備

`data/` 以下を年ごとに配置する（`--data_root` の既定値は `data`）。

```
data/
├── 2021/
│   ├── 1_<品種名>.ply
│   ├── 2_<品種名>.ply
│   ├── ...
│   └── yield_value.txt
├── 2022/
│   ├── ...
│   └── yield_value.txt
└── 2023/
    └── ...
```

### `.ply` ファイル

- ファイル名は `<plot_id>_<品種名>.ply`（`plot_id` は先頭の数字。品種名にアンダースコアを含んでも可）。
- vertex プロパティは `x, y, z` が必須。
  - `--use_rgb` を使う場合は `red, green, blue`（uchar 0–255）も必要。
  - `--use_normals` を使う場合は `nx, ny, nz` も必要。
- 各サンプルは `--num_point` 点（既定 4096）を**非復元抽出**する。点数が `--num_point` 未満のファイルが混入した場合は、
  警告を出したうえで復元抽出でパディングする。

### `yield_value.txt`

- 2 列 `品種名, 収量値`（収量の単位は g/m²）。
- 区切りは**タブ**または**2 個以上の連続スペース**。
- `品種` を含む行はヘッダ扱いでスキップ（実データではファイル末尾にある）。
- 空行は位置に関わらずスキップ。
- 収量値に含まれる桁区切りカンマは除去してから数値化する。

### `.ply` と収量の対応付け（ランクベース）

`plot_id` に欠番があり直接キーにできないため、
**`.ply` を `plot_id` の数値昇順に並べたもの**と、**`yield_value.txt` のデータ行を出現順に並べたもの**を、
位置（順位）で 1 対 1 に対応させる。

- `.ply` の個数と `yield_value.txt` のデータ行数が一致しないとエラーになる（欠番・重複を確認すること）。
- 品種名の表記ゆれ（全角／半角など）は警告のみで、対応関係は維持される。

---

## 学習（収量回帰）

### 最小実行

```bash
python train_regression.py --data_root data
```

### モデルの選択

| `--model` | 内容 |
| --- | --- |
| `pointnet_regression`（既定） | PointNet ベース。loss = MSE + feature transform 正則化 |
| `pointnet2_regression` | PointNet++ (MSG) ベース。loss = MSE |

```bash
python train_regression.py --model pointnet2_regression
```

### 追加入力チャネル

`.ply` の色・法線を xyz に連結して入力に加えられる（連結順は常に `xyz → rgb → normals`。入力チャネル数は 3 / 6 / 9）。

```bash
python train_regression.py --use_rgb --use_normals
```

### 年度ごと標準化

年度によって収量の水準が大きく異なるため、**既定では収量を年ごとに標準化**して loss を計算する
（「その年の中での相対的な高低」を学習させる狙い）。
`--no_standardize` を付けると生の g/m² スケールのまま loss を計算する（標準化前の baseline との比較用）。

> どちらの場合でも、**評価指標（MAE / RMSE / R²）は必ず生の g/m² スケールに戻して**報告される。

### 検証方式

- **単一分割**：`--split_ratio`（既定 0.9）で train/val を分割。年ごとに層化してから結合する。`--seed`（既定 42）固定。
- **k 分割交差検証**：`--n_folds 5` のように 2 以上を指定すると、1 回の実行で fold 0 〜 n_folds−1 を順に学習し、
  fold ごとの best val 指標を `cv_summary.csv` に集計する（このとき `--split_ratio` は無視される）。

```bash
python train_regression.py --n_folds 5
```

### CLI 引数一覧（[train_regression.py](train_regression.py) `parse_args()`）

| 引数 | 既定 | 説明 |
| --- | --- | --- |
| `--use_cpu` | False | CPU モードで実行 |
| `--gpu` | `0` | 使用 GPU（`CUDA_VISIBLE_DEVICES` に設定） |
| `--batch_size` | 8 | バッチサイズ |
| `--model` | `pointnet_regression` | モデル名（`models/` 配下のモジュール名） |
| `--epoch` | 100 | エポック数 |
| `--learning_rate` | 0.001 | 学習率（Adam） |
| `--num_point` | 4096 | 1 サンプルあたりの点数 |
| `--optimizer` | `Adam` | `Adam` 以外を指定すると SGD(lr=0.01, momentum=0.9) |
| `--log_dir` | None | 出力先ディレクトリ名（省略時はタイムスタンプ） |
| `--decay_rate` | 1e-4 | Adam の weight decay |
| `--lr_step_size` | 20 | StepLR の減衰間隔（epoch）※ |
| `--lr_gamma` | 0.7 | StepLR の減衰倍率 ※ |
| `--seed` | 42 | train/val 分割の乱数シード（train/val で同値必須） |
| `--data_root` | `data` | `data/{year}/` を含むルート |
| `--split_ratio` | 0.9 | train 分割比（学習データが少ないため 9:1 が既定） |
| `--no_standardize` | False | 収量の年度ごと標準化を無効化 |
| `--n_folds` | 1 | k 分割交差検証の fold 数（1 なら単一分割） |
| `--use_normals` | False | 法線 `nx,ny,nz` を入力に追加 |
| `--use_rgb` | False | 色 `r,g,b` を [0,1] 正規化して入力に追加 |

※ `--lr_step_size` / `--lr_gamma` の既定値は分類スクリプト（元は約 1 万件・200 epoch 用）からの流用で、
本タスクのデータ規模・epoch 数に合わせた検証はしていない。

---

## 出力物

`log/regression/{タイムスタンプ}/`（または `--log_dir` 名）以下に出力される。
まず実行時のモデルソース・`pointnet_utils.py`・`train_regression.py` がコピーされ、そのうえで:

### 単一分割の場合

```
log/regression/2026-08-31_12-00/
├── pointnet_regression.py         # 実行時ソースのコピー
├── pointnet_utils.py
├── train_regression.py
├── checkpoints/
│   └── best_model.pth             # val MAE が改善したときのみ保存
└── logs/
    ├── pointnet_regression.txt    # ログ
    ├── metrics.csv                # epoch ごとの train/val MAE・RMSE・R²
    ├── learning_curve.png         # MAE / RMSE / R² の 3 段プロット
    └── year_stats.csv             # 標準化に使った年ごとの平均・標準偏差
```

### k 分割交差検証の場合

```
log/regression/2026-08-31_12-00/
├── cv_summary.csv                 # fold ごとの best val 指標 + mean / std 行
├── fold_0/
│   ├── checkpoints/best_model.pth
│   └── logs/{metrics.csv, learning_curve.png, year_stats.csv, ...}
├── fold_1/
│   └── ...
└── ...
```

### `best_model.pth` の中身

| キー | 内容 |
| --- | --- |
| `epoch` | best（val MAE 最小）となった epoch |
| `val_mae` / `val_rmse` / `val_r2` | その時点の val 指標（g/m² スケール） |
| `standardized` | 標準化ありで学習されたか |
| `year_stats` | 標準化に使った年ごとの平均・標準偏差（推論時の逆標準化に必要） |
| `model_state_dict` / `optimizer_state_dict` | モデル・オプティマイザの状態 |

---

## 手法メモ

- **回帰ヘッド**：グローバル特徴 1024 → 512 → 256 → 1。出力に活性化を掛けず、生の float 値をそのまま予測値とする。
- **loss**：MSE。`pointnet_regression` ではこれに feature transform 正則化（`mat_diff_loss_scale=1e-3`,
  [models/pointnet_regression.py](models/pointnet_regression.py)）を加える。`pointnet2_regression` は素の MSE。
- **データ拡張**（学習時のみ、[utils/provider.py](utils/provider.py)）：
  `random_point_dropout` → `random_scale_point_cloud` → `shift_point_cloud` を xyz 座標にのみ適用。
  色・法線は正規化・拡張の対象外。
- **リーク対策**：年ごとの平均・標準偏差（`year_stats`）は **train 分割のみ**から計算し、val には train の値を渡して使う。

---

## 分類（ベースから継承）

ModelNet40 分類はベース実装のまま利用できる。データは `data/modelnet40_normal_resampled/` に配置する。

```bash
python train_classification.py
python test_classification.py --log_dir <run_name>
```

データセットの入手方法やオプションの詳細は
[yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) を参照すること。

---

## 既知の制約・TODO

- 収量回帰には `test_regression.py` 相当の推論専用スクリプトがなく、評価は学習中の val 指標のみ。
- `scheduler.step()` を `optimizer.step()` より前・epoch 先頭で呼び出しており（ベース由来）、PyTorch の警告が出る。
- `README.md` と `data/` は Git 管理外。

---

## クレジット / ライセンス

- ベース実装：[yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)（原著者 Benny / Xu Yan）
- ライセンス：MIT（[LICENSE](LICENSE)、Copyright (c) 2019 benny）
