# avif-mini-encoder

AVIF 向けの静止画エンコードに必要な部分だけを持つ、純 Rust の AV1 エンコーダー。libaom の次の設定を参考にしている:

- all-intra、1 フレームの静止画 (reduced still picture header)
- speed 5 (effort 4)
- tune IQ (アルファ面は tune PSNR のモノクロ)

libaom の出力とのバイト一致や、libaom と同等の速度・画質は目標にしていない。libaom と同じ設定で、実用になる画質と速度で AVIF を作れることが目標。

## 現状

### 動くもの

- **出力は libaom のデコーダー (aomdec) で正しく復号できる** (復号結果がエンコーダー側の再構成画像と一致する)。libaom のエンコーダーの出力とは一致しない。4:2:0 / 4:4:4 / モノクロ、8 bit、可逆 (qindex 0) も含む。
- 符号化ツール: 全イントラ予測モード (方向予測 + 角度差分、SMOOTH 系、PAETH、CfL、filter intra)、変換 (DCT / ADST / FLIPADST / IDTX / 1D、4x4〜64x64)、変換サイズ選択、量子化行列 (QM)、delta q (variance boost)、chroma delta q、デブロッキング、CDEF。
- tune IQ の調整 (QM レベル、chroma delta q、variance boost、sharpness、QM-PSNR 歪み、rdmult の重み、SSIM による rdmult スケーリング、CDEF adaptive) を参考に実装。
- x86-64 では主要な計算 (変換、予測、量子化、SATD、CDEF など) を実行時に AVX2 へ切り替える。`AV1_NO_AVX2=1` で通常経路を強制できる。
- arm64 では NEON (ベースラインなので実行時検出は不要) の手書きカーネル (SATD、CfL、角度予測の補間、係数コスト文脈) を使う。`AV1_NO_NEON=1` で通常経路を強制できる。出力は AVX2 / NEON / 通常経路でビット一致する。
- CDEF 強度探索の並列化と、色・アルファの同時エンコード。出力はスレッド数によらずバイト一致。

### 未実装・既知の差分

- RGB 変換と AVIF コンテナの組み立てを含む API は未実装。公開 API は YUV プレーン → AV1 OBU まで。
- 1 タイルのみ。幅 4096 / 面積 4096×2304 を超える画像は `encode_av1()` がエラーを返す。
- palette / intrabc、HORZ_A/B・VERT_A/B・HORZ_4/VERT_4 分割は未実装。
- libaom の ML / CNN による分割の枝刈りは使わず、単純なヒューリスティックで代用している。
- 同じ qindex で、libaom と画質・サイズが異なる (画像による)。

## 使い方

```rust
use avif_mini_encoder::{encode_av1, Chroma, ColorDesc, Tune};

// planes: [Y, U, V] (4:2:0 なら U/V は (w+1)/2 × (h+1)/2)
let color = Some(ColorDesc { primaries: 1, transfer: 13, matrix: 6 });
let (obu, recon) = encode_av1(&planes, w, h, Chroma::Yuv420, color, true, qindex, Tune::Iq, 5)?;
```

- `encode_av1_threads`: 並列化できる段階にスレッドを使う。
- `encode_av1_pair`: 色とアルファを同時にエンコードする。
- 戻り値は AV1 OBU と再構成プレーン。

PPM を読んで OBU を書き出す検証ツールが `examples/rawenc.rs`。

## 構成

| パス | 内容 |
|---|---|
| `src/ec.rs` | 多シンボル算術符号器、ビットライター |
| `src/cdf.rs` | CDF の構造体・既定値・適応・コスト |
| `src/obu.rs` | シーケンスヘッダー / フレームヘッダー OBU |
| `src/consts.rs` | ブロック / 変換サイズ、モード等の定数 |
| `src/txfm/` | 2D 順変換・逆変換 (`txfm1d.rs` は libaom の 1D 変換を機械変換したもの) |
| `src/predict.rs`, `src/avail.rs` | イントラ予測と隣接ブロックの可用性 |
| `src/quant.rs`, `src/coef.rs` | 量子化 (QM 対応) / 係数の文脈・書き込み・レート |
| `src/enc/` | フレーム全体の流れ、RD 探索、ブロック符号化、構文書き出し |
| `src/lf.rs`, `src/cdef.rs` | デブロッキング / CDEF と強度探索 |
| `src/tables/` | libaom から生成した表 (CDF 既定値、量子化表、QM、スキャン、サイズ表など) |
| `examples/` | `rawenc` (検証・測定用)、`pairbench` (色 + アルファの同時エンコードの比較) |
| `tools/` | 評価・比較スクリプト (`kodak-eval.py`、`bench-compare.py`) |

## 探索の概要 (libaom speed 5 との対応)

- **分割**: NONE → SPLIT → HORZ/VERT の順。大きいブロックは正方形のみ (libaom の `use_square_partition_only_threshold`)。HORZ/VERT は SPLIT の RD が NONE 以下のときだけ試す。
- **輝度モード**: 13 モード + filter intra 5 種を SATD モデルで順位付けし、上位 3 つを RD で評価。方向予測は上位 2 モードだけ角度差分を追加。勝者で変換サイズ × 変換タイプ (libaom の reduced intra set) を探索。
- **色差**: 13 モードを SATD で順位付けした上位 3 + CfL (α は SSE で U/V 独立に決定) を RD で評価。
- **歪み**: 変換領域 (QM 重み付き)。64 点変換を含むサイズは、係数が半分捨てられるので画素領域 (libaom と同じ)。

## 開発

`cargo test --release` で単体テスト (変換の往復、AVX2 / NEON と通常経路の一致など) が走る。

libaom (aomdec) との一致確認:

```sh
cargo build --release --example rawenc
target/release/examples/rawenc in.ppm <qindex> out 420     # 420 | 444 | mono
aomdec --rawvideo -o out.dec.yuv out.obu && cmp out.dec.yuv out.rec.yuv
```

変更前後の速度比較は `python3 tools/bench-compare.py BEFORE AFTER in.ppm` (出力の完全一致も検査する)。

`tools/kodak-eval.py` は numpy、scipy、pillow、ssimulacra2 を使う (`pip install numpy scipy pillow ssimulacra2`)。

## ライセンス

BSD-2-Clause ([LICENSE](LICENSE))。
