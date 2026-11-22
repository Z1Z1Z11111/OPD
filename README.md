# ALFWorld ReAct & GRPO 基线

本项目面向 **ALFWorld** 文本交互环境，提供统一的 **ReAct / GRPO** 二选一评测与训练框架：

- **`method: react`**：使用本地离线 vLLM 与 HiPER 风格 `<think>/<action>` ReAct 模板评测基础模型；
- **`method: grpo`**：在完全一致的 ReAct 交互协议下通过 GRPO 强化学习训练并评测导出权重。

> 本仓库仅保留 ALFWorld ReAct + FSDP + vLLM/HF GRPO 所需的最小实现，已移除 HiPER 的 Plan–Execute（Options）、HAE、GiGPO、其余 benchmark 环境、Megatron/SGLang 后端以及旧版 Raw JSON ReAct / Transformers 评测代码。

## 目录结构

```text
OPD/
├─ configs/
│  ├─ alfworld.yaml        # 统一主配置（method: react | grpo、模型、采样、历史长度、vLLM 参数）
│  └─ grpo.yaml            # GRPO 训练超参覆盖表
├─ agent.py                # HiPER ReAct 模板、最近窗口历史、动作投影与协程主循环
├─ model.py                # 本地离线 vLLM 推理封装与 token 统计
├─ alfworld_env.py         # ALFWorld 环境适配器（支持多卡分片 + 单卡合批分片）
├─ evaluate.py             # 单卡评测入口（串行或合批，支持 --method react / grpo）
├─ batched_evaluate.py     # 单卡多任务动态合批调度
├─ parallel_evaluate.py    # 多卡多进程并行评测与汇总
├─ report.py               # 六类任务统计与 PNG/SVG 绘图
├─ prepare_data.py         # ALFWorld 数据下载、导入与校验
├─ train-rl                # GRPO 正式训练启动脚本
├─ rl/
│  ├─ __init__.py          # react / grpo 模式切换与 checkpoint 解析
│  ├─ train.py             # GRPO 训练入口（支持 --dry-run / --smoke）
│  ├─ progress.py          # 终端实时跟踪 train.log 进度
│  └─ hiper/               # 精简后的最小 veRL + ALFWorld ReAct GRPO 框架
└─ tests/                  # 无 GPU mock 回归测试
```

## 快速开始

### 1. 环境与数据准备

推荐 Linux + Python 3.11 + CUDA GPU：

```bash
conda create -n alfworld-vllm python=3.11 -y
conda activate alfworld-vllm
python -m pip install 'cmake>=3.18,<4'
python -m pip install -r requirements.txt

# 校验本地 ALFWorld 数据（train 3553 / valid_seen 140 / valid_unseen 134）
python prepare_data.py --verify-only
```

### 2. ReAct 基线评测（不训练）

```bash
# 单卡评测 valid_seen
python evaluate.py --method react --split seen

# 单卡全量 seen + unseen
python evaluate.py --method react --split both

# 多卡并行 + 每卡 4 任务合批
python parallel_evaluate.py --gpus 0,1,2 --method react --split both --batch-size 4
```

### 3. GRPO 训练与评测

默认使用双卡 vLLM（`tensor_model_parallel_size=2`），每批 `16` 个任务 × 每任务 `8` 条轨迹，学习率 `1e-6`，KL 系数 `0.01`，无效格式惩罚 `0.1`，episode 奖励：

```bash
# 打印最终解析出的训练参数（不启动训练）
python -m rl.train --dry-run

# 单卡最小冒烟检查（1 步更新，4 条轨迹 × 2 步，结果指针写入 smoke.json）
python -m rl.train --smoke --model models/Qwen2.5-1.5B-Instruct

# 正式 GRPO 训练（日志写入 outputs/grpo/launch-*/train.log）
./train-rl

# 另开终端监控训练进度
python -m rl.progress
```

训练完成后会自动导出 Hugging Face 权重并更新 `outputs/grpo/<基础模型>/latest.json`，随后即可直接切换 `--method grpo` 评测：

```bash
# 评测最新导出的 GRPO 权重
python evaluate.py --method grpo --split seen
python parallel_evaluate.py --gpus 0,1,2 --method grpo --split both --batch-size 4

# 或显式指定某一步导出的权重目录
python evaluate.py --method grpo --checkpoint outputs/grpo/.../global_step_N/actor/huggingface --split unseen
```

也可以直接修改 [configs/alfworld.yaml](configs/alfworld.yaml) 中的 `method: react` 或 `method: grpo`。

### 4. 统计重绘与回归测试

```bash
# 对已有运行目录独立重新生成图表
python report.py --run-dir outputs/<运行目录>

# 运行全部无 GPU 单元测试
python -m unittest discover -s tests -v
```

## 交互协议要点

- 每轮构造单条 `user` 消息，默认携带最近 `history_length=2` 步的“执行前观测 + 动作”，不向后续提示词回填 `<think>`（完整 thought 保存在 `events.jsonl` 与 `result.json`）。
- 提示词剔除 `help` 动作；模型输出按 `<think>...</think><action>...</action>` 提取并转小写；缺失 `<action>` 标签时回退为小写输出的最后 30 个字符；每轮严格消耗 1 个环境步，不做重试。
- `is_action_valid` 校验 `<think>`/`<action>` 标签完整且不含中文字符，在 GRPO 训练中用于施加格式惩罚。

许可证与上游致谢见 [LICENSE](LICENSE) 与 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
