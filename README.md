# DualTSR_Repro

这是 **DualTSR: Unified Dual-Diffusion Transformer for Scene Text Image Super-Resolution** 的 PyTorch 复现工程。主干基于 AMD Nitro-E 发布的 **E-MMDiT** 结构，并针对 DualTSR 改造成图像速度与文本 token 双输出模型。

当前实现覆盖论文中的核心流程：

- 使用 Conditional Flow Matching 生成高分辨率图像 latent。
- 使用 absorbing-state 离散扩散建模文本 token。
- 使用 MM-DiT 风格的图文联合注意力，让图像流和文本流在每层交互。
- 使用 EMA teacher 构造 model-guided 图像速度场训练目标。
- 推理时使用 Euler 图像采样，并同步执行文本逐步 unmask。
- 文本扩散序列使用显式 EOS；论文的 24 字符上限对应 25 个 token（含 EOS）。
- 使用 E-MMDiT 的 AdaLN-affine、多路径 token 压缩/重建、Position Reinforcement 和 Alternating Subregion Attention。

## 环境安装

```bash
pip install -r requirements.txt
```

如果需要在昇腾 NPU 上训练，还需要安装与本机 CANN/PyTorch 版本匹配的 `torch_npu`。

## 数据准备

论文使用 CTR scene 图像构建 CTR-TSR，过滤规则如下：

- 图像长边不小于 64。
- 宽高比大于 2。
- 文本长度不超过 24。
- HR 统一 resize 到 128 x 512。
- 训练时在线生成 LR，退化流程采用 BSRGAN/Real-ESRGAN 风格的 blind degradation。

从官方 CTR LMDB 生成词表和可选 manifest：

```bash
python3 scripts/prepare_ctr_tsr.py --config configs/train/dualtsr_ctr_4x.yaml
```

正式训练前，需要先修改 `configs/train/dualtsr_ctr_4x.yaml` 中的 LMDB 和 TransOCR 占位路径。E-MMDiT 默认使用 Nitro-E 官方采用的公开 DC-AE，首次启动会从 Hugging Face 下载约 1.25 GB 权重。

可在启动分布式训练前执行严格预检：

```bash
python3 scripts/check_reproduction_ready.py \
  --config configs/train/dualtsr_ctr_4x.yaml \
  --world-size 4 \
  --stage train
```

预检会验证 LMDB、词表、VAE/OCR 权重、global batch/梯度累积设置及 E-MMDiT 参数规模。CTR 数据和 TransOCR 基线见 [FudanVI 官方仓库](https://github.com/FudanVI/benchmarking-chinese-text-recognition)。默认视觉 tokenizer 是 Nitro-E 使用的公开 [DC-AE f32c32](https://huggingface.co/mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers)；仓库仍保留 RDP VAE 兼容入口，但公开检索未找到对应权重。

TransOCR 权重可从 FudanVI 官方 Google Drive 文件夹按名称筛选下载：

```bash
python3 scripts/download_transocr_assets.py --list-only
python3 scripts/download_transocr_assets.py --output weights/transocr
```

完整评估前使用 `--stage evaluate` 或 `--stage all` 再做一次预检。

## 预训练数据合成

在 CTR-TSR 之外，仓库提供一条字体渲染的合成管线，用于在真实数据前先做预训练。

第一步，下载开源字体（全部 SIL OFL 1.1，共 13 款：思源黑体/宋体、霞鹜文楷、站酷系列、马善政等手写体、Lato）：

```bash
python3 scripts/download_fonts.py        # 下载到 assets/fonts/，并生成 fonts.json 清单
```

合成字符集默认取「至少被 6/13 款字体覆盖」的字符（约 6800 常用汉字 + ASCII + 常见标点），
可通过 `synth.charset_min_fonts` 调整。渲染随机化包括字体、字号、字距、基线抖动、
颜色（保证与背景的亮度对比）、描边、阴影、纯色/渐变/噪声/照片背景、旋转和透视扰动；
LR 不落盘，训练时按 BSRGAN/Real-ESRGAN 风格在线退化。

两种使用方式：

```bash
# 方式一：在线渲染（推荐，数据无限）。先生成词表，再直接训练：
python3 scripts/synthesize_pretrain_data.py --config configs/train/dualtsr_pretrain_synth.yaml --vocab-only
torchrun --nproc_per_node=4 train.py --config configs/train/dualtsr_pretrain_synth.yaml

# 方式二：离线落盘为 CTR 格式 LMDB（或 --format images 输出 manifest+图片）：
python3 scripts/synthesize_pretrain_data.py --config configs/train/dualtsr_pretrain_synth.yaml
```

离线产物与官方 CTR LMDB 键格式一致（`num-samples`/`image-%09d`/`label-%09d`），
可直接用 `data.type: ctr_lmdb` 读取。可选参数：`--corpus`（每行一条文本的语料，
替代随机字符串采样）、`--bg-dir`（真实照片背景目录）、`--workers`、`--seed`。

CPU 冒烟验证全链路：

```bash
python3 scripts/synthesize_pretrain_data.py --config configs/train/smoke_synth.yaml --vocab-only
python3 train.py --config configs/train/smoke_synth.yaml
```

## 非 Colab 在线合成训练入口

服务器、本地 CUDA、AutoDL/SeeTaCloud 等非 Colab 环境可直接使用：

```bash
pip install -r requirements.txt
bash scripts/run_online_synth_train.sh
```

入口配置为 `configs/train/online_synth_emmdit.yaml`，默认使用在线字体渲染合成数据：

- 自动下载开源字体到 `assets/fonts/`。
- 只生成 `data/online_synth/vocab.txt`，训练时在线渲染 HR 文本图并在线退化得到 LQ，不预生成 LMDB。
- 自动下载 DC-AE 基础权重到 `weights/dc-ae-f32c32-sana-1.0-diffusers`。
- 默认用 `--resume auto`，会从 `outputs/online_synth_emmdit/checkpoints/latest.pt` 自动断点续训。

`scripts/download_fonts.py` 输出的 `N/13 fonts ready` 只表示内置下载清单的完成情况；
训练实际使用 `assets/fonts/` 顶层所有 `.ttf/.otf/.ttc` 字体。启动脚本会额外打印
`training font pool: ... files`，以这个数量为准。

常用覆盖项：

```bash
# 单卡 4090 类环境，提高显存利用率
BATCH_SIZE=128 GLOBAL_BATCH_SIZE=128 LR=0.0003 bash scripts/run_online_synth_train.sh

# 多卡
NPROC_PER_NODE=4 BATCH_SIZE=64 GLOBAL_BATCH_SIZE=256 bash scripts/run_online_synth_train.sh

# 自定义输出目录、步数和保存间隔
RUN_NAME=online_synth_fast MAX_STEPS=20000 SAVE_EVERY=500 bash scripts/run_online_synth_train.sh
```

## Colab 训练入口

Colab 入口位于 `notebooks/DualTSR_EMMDiT_Colab.ipynb`。notebook 会完成：

- 克隆仓库并安装 `requirements.txt`。
- 可选挂载 Google Drive，并把 `outputs/<run_name>/checkpoints/latest.pt` 写到 Drive，支持 Colab 断开后继续训练。
- 自动下载开源字体并生成合成预训练词表。
- 自动下载 Nitro-E 使用的公开 DC-AE 基础权重到 `weights/dc-ae-f32c32-sana-1.0-diffusers`。
- 生成 `configs/train/colab_runtime.yaml`，再用 `python3 train.py --config configs/train/colab_runtime.yaml --resume auto` 启动或恢复训练。

默认配置 `configs/train/colab_emmdit_synth.yaml` 使用在线字体渲染数据和较小的 E-MMDiT 宽度，方便单张 Colab GPU 启动训练。若要使用 CTR 数据，在 notebook 中设置：

```python
DATASET = "ctr"
CTR_URL = "你的 Google Drive 文件夹或压缩包链接"
```

也可以直接填 `CTR_TRAIN_LMDB` 和 `CTR_VAL_LMDB`。准备脚本会下载/解压 CTR 数据、定位 LMDB、生成词表，然后写入运行时配置：

```bash
python3 scripts/colab_prepare.py \
  --dataset synth \
  --drive-root /content/drive/MyDrive/DualTSR_Repro \
  --run-name colab_emmdit_synth
```

## 模型替换入口

为了后续替换 MMDiT、VAE 和 TextEncoder，当前代码把这三块都收敛到配置化适配器：

- VAE：入口在 `dualtsr/vae/__init__.py` 的 `build_vae()`。支持 `IdentityVAE`、diffusers `AutoencoderKL`、RDP VAE 和自定义模块。
- TextEncoder：入口在 `dualtsr/model.py` 的 `build_text_encoder()`。内置 `char`，可用 `custom` 接外部文本编码器。
- MMDiT：正式配置使用 `dualtsr.emmdit:EMMDiTBackbone`；`NativeMMDiTBackbone` 仅作为轻量基线和基础设施 smoke test。

论文规模配置：

```yaml
vae:
  type: autoencoder_dc
  pretrained_path: weights/dc-ae-f32c32-sana-1.0-diffusers
  latent_channels: 32
  latent_size: [4, 16]
  scaling_factor: 0.41407

model:
  hidden_dim: 768
  text_encoder:
    type: char
  mmdit:
    class_path: dualtsr.emmdit:EMMDiTBackbone
    init_args:
      patch_size: 1
      num_heads: 24
      group_depths: [4, 16, 4]
      mlp_ratio: 3.0
      use_subregion_attention: true
```

替换为自定义实现时，只需要提供可 import 的类路径：

```yaml
vae:
  type: custom
  class_path: my_project.models:MyVAE
  latent_channels: 4
  latent_size: [16, 64]
  scaling_factor: 1.0
  kwargs:
    checkpoint: /path/to/vae.pt

model:
  hidden_dim: 768
  text_encoder:
    type: custom
    class_path: my_project.models:MyTextEncoder
    output_dim: 1024
    trainable: true
    kwargs:
      checkpoint: /path/to/text_encoder.pt
  mmdit:
    type: custom
    class_path: my_project.models:MyMMDiT
    trainable: true
    kwargs:
      checkpoint: /path/to/mmdit.pt
```

自定义类接口约定如下：

- 自定义 VAE 继承 `torch.nn.Module`，实现 `encode(image) -> latent` 和 `decode(latent) -> image`；实际 latent 形状由启动时 dry-run 自动推断。
- 自定义 TextEncoder 继承 `torch.nn.Module`，实现 `forward(text_tokens, batch_size, max_length, device) -> embeddings`，输出形状为 `[B, max_length, output_dim]`。当 `output_dim != model.hidden_dim` 时，主模型会自动加一层线性投影。
- 自定义 MMDiT 继承 `torch.nn.Module`，实现 `forward(x_img, timesteps, text_embeddings, lr=None) -> (velocity, text_embeddings)`。

## 训练

CPU smoke 测试：

```bash
python3 train.py --config configs/train/smoke.yaml
python3 train.py --config configs/train/smoke_emmdit.yaml
python3 train.py --config configs/train/smoke_emmdit_dcae.yaml
python3 train.py --config configs/train/smoke_resume.yaml --resume auto
```

`smoke_emmdit.yaml` 会实际经过 E-MMDiT 的压缩、ASA、重建和双输出路径；`smoke_emmdit_dcae.yaml` 进一步使用下载后的真实 32× DC-AE；`smoke.yaml` 保留轻量原生 MMDiT，便于快速检查通用训练基础设施。

论文规模训练示例：

```bash
torchrun --nproc_per_node=4 train.py --config configs/train/dualtsr_ctr_4x.yaml
```

使用昇腾 NPU 时，在 YAML 中设置：

```yaml
runtime:
  device: npu
  precision: bf16
```

分布式 backend 会自动选择：CUDA 使用 NCCL，NPU 使用 HCCL，CPU 使用 Gloo。

## 推理

```bash
python3 infer.py --config configs/infer/smoke.yaml
python3 infer.py --config configs/train/smoke_emmdit.yaml
python3 infer.py --config configs/train/online_synth_emmdit.yaml \
  --checkpoint outputs/online_synth_emmdit/checkpoints/latest.pt \
  --input /path/to/lq_images \
  --output outputs/online_synth_emmdit/infer

# 多卡并行推理:每个 rank 自动处理一部分输入,rank0 合并 predictions.*
torchrun --nproc_per_node=4 infer.py --config configs/train/online_synth_emmdit.yaml \
  --checkpoint outputs/online_synth_emmdit/checkpoints/latest.pt \
  --input /path/to/lq_images \
  --output outputs/online_synth_emmdit/infer
```

输出目录会包含 `images/`、`predictions.jsonl` 和 `predictions.csv`。默认会把 DualTSR 文本路预测结果叠加到 SR 图像左上角；可用 `--set infer.overlay_text=false` 关闭。

## 评估

```bash
python3 evaluate.py --config configs/train/dualtsr_ctr_4x.yaml
```

PSNR 已内置实现。LPIPS/FID 会在可选依赖和对应模型权重可用时启用。论文协议中的 ACC/NED 来自固定 TransOCR 对 SR 图像的识别结果：先生成包含 `id,text` 的 JSONL/CSV，并配置 `evaluation.ocr_predictions`。若未提供，评估脚本会回退到 DualTSR 内部文本输出，但会明确标记为诊断指标而非论文协议。
