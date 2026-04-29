# CogVideoX Cat LoRA

这是一个围绕猫咪视频数据集做 `CogVideoX LoRA` 微调、对比生成与测评的小型工程。仓库里已经包含：

- 文生视频 LoRA 训练脚本：[`train_cogvideox_lora.py`](/Users/man.tang/PycharmProjects/cogvideox-cat-lora/train_cogvideox_lora.py)
- 图生视频 LoRA 训练脚本：[`train_cogvideox_image_to_video_lora.py`](/Users/man.tang/PycharmProjects/cogvideox-cat-lora/train_cogvideox_image_to_video_lora.py)
- Base/LoRA 对比生成脚本：[`compare_base_lora.py`](/Users/man.tang/PycharmProjects/cogvideox-cat-lora/compare_base_lora.py)
- Gradio 演示：[`gradio_app.py`](/Users/man.tang/PycharmProjects/cogvideox-cat-lora/gradio_app.py)
- 自动测评与人工复核工具：[`evaluation/`](/Users/man.tang/PycharmProjects/cogvideox-cat-lora/evaluation)

相关参考链接：

- CogVideoX 项目：[THUDM/CogVideo](https://github.com/THUDM/CogVideo)
- Diffusers 中的 CogVideoX 训练与推理生态：[huggingface/diffusers](https://github.com/huggingface/diffusers)
- LoRA 论文：[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- VBench 项目：[VBench](https://github.com/Vchitect/VBench)

## 项目结构

```text
cogvideox-cat-lora/
├── dataset/
│   ├── prompts.txt
│   ├── videos.txt
│   └── test_prompts.txt
├── evaluation/
│   ├── videobench_lite_eval.py
├── compare_base_lora.py
├── gradio_app.py
├── train_cogvideox_lora.py
├── train_cogvideox_image_to_video_lora.py
└── requirements.txt
```

## 环境准备

建议使用 Python 3.10 到 3.12，并提前准备好 CUDA 环境与可用显卡。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

训练脚本依赖较新的 `diffusers`，建议直接安装源码版：

```bash
git clone https://github.com/huggingface/diffusers.git
cd diffusers
pip install -e .
cd ..
```

然后初始化 `accelerate`：

```bash
accelerate config default
```

## 数据组织

当前仓库默认使用两文件格式：

- [`dataset/prompts.txt`](/Users/man.tang/PycharmProjects/cogvideox-cat-lora/dataset/prompts.txt)：逐行文本提示词
- [`dataset/videos.txt`](/Users/man.tang/PycharmProjects/cogvideox-cat-lora/dataset/videos.txt)：逐行视频相对路径

目录示例：

```text
dataset/
├── prompts.txt
├── videos.txt
└── videos/
    ├── 001.mp4
    ├── 002.mp4
    └── ...
```

注意事项：

- `videos.txt` 里的路径必须相对 `--instance_data_root`。
- `prompts.txt` 和 `videos.txt` 的行数必须一一对应。
- `dataset/test_prompts.txt` 用于推理对比和测评，不参与训练。
- 建议视频尽量统一分辨率、帧数和时长，能明显减少训练时的数据预处理问题。

## 模型微调方法

这里的训练主线保持不变：使用 `Diffusers + PEFT` 对 `CogVideoX` 做 `LoRA` 微调，而不是全量微调。

### 1. 文生视频 LoRA 微调

最小可执行命令如下：

```bash
accelerate launch train_cogvideox_lora.py \
  --pretrained_model_name_or_path THUDM/CogVideoX-2b \
  --instance_data_root dataset \
  --caption_column prompts.txt \
  --video_column videos.txt \
  --validation_prompt "A vertical close-up shot of an orange tabby cat resting on a soft indoor cushion in warm natural sunlight." \
  --num_validation_videos 1 \
  --validation_epochs 5 \
  --seed 42 \
  --rank 64 \
  --lora_alpha 64 \
  --mixed_precision bf16 \
  --output_dir outputs/cogvideox-cat-lora \
  --height 480 \
  --width 720 \
  --fps 8 \
  --max_num_frames 49 \
  --train_batch_size 1 \
  --num_train_epochs 30 \
  --checkpointing_steps 500 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 200 \
  --enable_slicing \
  --enable_tiling \
  --optimizer AdamW \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0
```

常用参数说明：

- `--pretrained_model_name_or_path`：基础模型，可以是本地路径或 Hugging Face 模型名。
- `--instance_data_root`：数据根目录，这个仓库默认就是 `dataset`。
- `--caption_column prompts.txt`：提示词文件名。
- `--video_column videos.txt`：视频列表文件名。
- `--rank` 和 `--lora_alpha`：LoRA 规模，当前项目更推荐从 `32/64` 或 `64/64` 起步。
- `--height --width --fps --max_num_frames`：训练前的视频采样与重整形策略。
- `--validation_prompt`：训练中用于定期抽样验证的提示词。
- `--output_dir`：LoRA 权重、checkpoint、验证视频输出目录。

经验建议：

- 当前脚本最稳妥的起点仍然是 `CogVideoX-2b`。
- 如果数据提示词质量一般，`rank=64` 往往比很低的 rank 更稳。
- 25 到 50 条同风格视频可以先做概念验证，正式训练建议准备更多样本。
- 训练中请务必打开验证采样，不然很难及时发现过拟合或退化。

### 2. 图生视频 LoRA 微调

如果你要做 `Image-to-Video` 微调，使用下面的脚本：

```bash
accelerate launch train_cogvideox_image_to_video_lora.py \
  --pretrained_model_name_or_path THUDM/CogVideoX-5b-I2V \
  --instance_data_root dataset \
  --caption_column prompts.txt \
  --video_column videos.txt \
  --validation_prompt "A fluffy cat slowly turns its head in warm window light." \
  --validation_images "assets/val_cat.png" \
  --num_validation_videos 1 \
  --validation_epochs 5 \
  --seed 42 \
  --rank 64 \
  --lora_alpha 64 \
  --mixed_precision bf16 \
  --output_dir outputs/cogvideox-cat-i2v-lora \
  --height 480 \
  --width 720 \
  --fps 8 \
  --max_num_frames 49 \
  --train_batch_size 1 \
  --num_train_epochs 30 \
  --learning_rate 1e-4
```

这里和文生视频的核心区别只有两点：

- 使用 `train_cogvideox_image_to_video_lora.py`
- 额外提供 `--validation_images`，并保证图片顺序与 `--validation_prompt` 一一对应

## 推理与对比生成方法

训练完成后，可以把同一组测试提示词分别送入 base 模型和 LoRA 模型，生成对比视频：

```bash
python3 compare_base_lora.py \
  --base_model THUDM/CogVideoX-2b \
  --lora_path outputs/cogvideox-cat-lora \
  --prompts_file dataset/test_prompts.txt \
  --output_root compare_outputs \
  --height 960 \
  --width 544 \
  --num_frames 25 \
  --num_inference_steps 50 \
  --guidance_scale 6 \
  --fps 8 \
  --lora_alpha 64 \
  --lora_rank 64
```

输出目录会长这样：

```text
compare_outputs/
├── base/
│   ├── 001.mp4
│   ├── 002.mp4
│   └── ...
└── lora/
    ├── 001.mp4
    ├── 002.mp4
    └── ...
```

如果想开一个本地 Web 界面手动试玩：

```bash
python3 gradio_app.py \
  --base_model THUDM/CogVideoX-2b \
  --lora_path outputs/cogvideox-cat-lora \
  --output_dir gradio_outputs \
  --lora_scale 1.0 \
  --server_port 7860
```

## 模型测评方法

这个仓库现在只保留一条主测评路径：`Video-Bench-lite`。

### Video-Bench-lite 语义测评

[`evaluation/videobench_lite_eval.py`](/Users/man.tang/PycharmProjects/cogvideox-cat-lora/evaluation/videobench_lite_eval.py) 是一个面向本仓库目录结构的轻量化评测脚本。它参考了 [VBench](https://github.com/Vchitect/VBench) 将视频评估拆成多个维度的思路，但不是官方 VBench 的原样复现；当前实现更适合用于 `CogVideoX base` 和 `CogVideoX LoRA` 的成对横向比较。

当前保留的核心思路：

- `visual_score`：清晰度、对比度、饱和度、曝光
- `temporal_score`：时序一致性与光流平滑性
- `prompt_score`：基于 CLIP 的图文语义一致性
- `total_score`：上述三类分数的加权汇总

适用场景：

- 比较同一组 prompt 下 base 与 LoRA 的相对提升
- 做训练回合之间的横向实验记录
- 辅助人工复核，而不是替代人工判断

```bash
python3 evaluation/videobench_lite_eval.py \
  --compare_dir compare_outputs \
  --prompts_file dataset/test_prompts.txt
```

主要输出：

- `compare_outputs/videobench_lite_results/scores.csv`
- `compare_outputs/videobench_lite_results/pair_summary.csv`
- `compare_outputs/videobench_lite_results/report.json`
- `compare_outputs/videobench_lite_results/report.md`
- `compare_outputs/videobench_lite_results/review.html`

## 推荐工作流

如果你想从零到一完整跑一轮，建议直接按下面顺序：

```bash
# 1) 训练 LoRA
accelerate launch train_cogvideox_lora.py ...

# 2) 生成 base / lora 对比视频
python3 compare_base_lora.py \
  --base_model THUDM/CogVideoX-2b \
  --lora_path outputs/cogvideox-cat-lora \
  --prompts_file dataset/test_prompts.txt \
  --output_root compare_outputs

# 3) 跑 Video-Bench-lite
python3 evaluation/videobench_lite_eval.py \
  --compare_dir compare_outputs \
  --prompts_file dataset/test_prompts.txt
```

## 当前仓库的已知特点

- 训练脚本本身很完整，但文档此前还是上游示例，和本仓库实际目录不匹配。
- 评测工具偏工程化实用，其中 `videobench_lite_eval.py` 明确是参考 `VBench` 思路的轻量化实现，不是官方基准复现，适合做自己实验的横向对比。
- 当前项目已经去掉旧的多分支评测脚本，只保留一条更清晰的 `CogVideoX LoRA + VBench-like` 主流程。

## 排错建议

- `CUDA out of memory`：先降 `--height`、`--width`、`--max_num_frames` 或开启更小 batch。
- `validation` 视频不出结果：先检查 `--validation_prompt` 是否传入，以及输出目录是否可写。
- LoRA 效果不明显：优先检查 `lora_alpha / rank` 比例、训练步数、提示词质量和测试提示词是否过于偏离训练分布。
- 测评脚本找不到文件：确认 `compare_outputs/base` 和 `compare_outputs/lora` 下的命名是否是一一对应的 `001.mp4`、`002.mp4` 形式。
