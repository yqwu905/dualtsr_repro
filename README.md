# DualTSR_Repro

这是 **DualTSR: Unified Dual-Diffusion Transformer for Scene Text Image Super-Resolution** 的 PyTorch 复现工程骨架。

当前实现覆盖论文中的核心流程：

- 使用 Conditional Flow Matching 生成高分辨率图像 latent。
- 使用 absorbing-state 离散扩散建模文本 token。
- 使用 MM-DiT 风格的图文联合注意力，让图像流和文本流在每层交互。
- 使用 EMA teacher 构造 model-guided 图像速度场训练目标。
- 推理时使用 Euler 图像采样，并同步执行文本逐步 unmask。

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

正式训练前，需要先修改 `configs/train/dualtsr_ctr_4x.yaml` 中的 LMDB、VAE 和 TransOCR 占位路径。

## 模型替换入口

为了后续替换 MMDiT、VAE 和 TextEncoder，当前代码把这三块都收敛到配置化适配器：

- VAE：入口在 `dualtsr/vae.py` 的 `build_vae()`。内置 `identity`、`autoencoder_kl` 和 `custom`。
- TextEncoder：入口在 `dualtsr/model.py` 的 `build_text_encoder()`。内置 `char`，可用 `custom` 接外部文本编码器。
- MMDiT：入口在 `dualtsr/model.py` 的 `build_mmdit()`。内置 `native`，可用 `custom` 接外部 MM-DiT 主干。

默认配置仍兼容旧写法；推荐新配置把 TextEncoder 和 MMDiT 显式写在 `model` 下面：

```yaml
vae:
  type: autoencoder_kl
  pretrained_path: /path/to/pretrained/vae
  latent_size: [16, 64]
  scaling_factor: 0.18215

model:
  patch_size: [2, 2]
  hidden_dim: 768
  text_encoder:
    type: char
  mmdit:
    type: native
    depth: 12
    num_heads: 12
    mlp_ratio: 4.0
    dropout: 0.0
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

- 自定义 VAE 继承 `torch.nn.Module`，实现 `encode(image) -> latent` 和 `decode(latent) -> image`；如果类没有 `info.latent_channels/info.latent_size`，需要在 YAML 中填写 `latent_channels` 和 `latent_size`。
- 自定义 TextEncoder 继承 `torch.nn.Module`，实现 `forward(text_tokens, batch_size, max_length, device) -> embeddings`，输出形状为 `[B, max_length, output_dim]`。当 `output_dim != model.hidden_dim` 时，主模型会自动加一层线性投影。
- 自定义 MMDiT 继承 `torch.nn.Module`，实现 `forward(img_tokens, text_tokens, timesteps) -> (img_tokens, text_tokens)`，输入输出 token 维度都应为 `model.hidden_dim`。

## 训练

CPU smoke 测试：

```bash
python3 train.py --config configs/train/smoke.yaml
python3 train.py --config configs/train/smoke_resume.yaml --resume auto
```

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
```

输出目录会包含 `images/`、`predictions.jsonl` 和 `predictions.csv`。

## 评估

```bash
python3 evaluate.py --config configs/train/dualtsr_ctr_4x.yaml
```

PSNR 已内置实现。LPIPS/FID 会在可选依赖和对应模型权重可用时启用。ACC/NED 会基于预测文本和 GT 文本字段计算；外部 TransOCR 评估保持为可配置路径，因为本仓库不内置 TransOCR 代码和权重。
