# ALFWorld 新方法开发

默认配置现为 **HiPER 普通 ReAct + 本地 vLLM 推理**。
这是推理评测入口；强化学习训练、策略损失和训练权重同步尚未接入。
原全历史 JSON ReAct 配置保存在 `configs/alfworld_raw_transformers.yaml`。

主要源码如下，report.py 负责统计绘图：

```text
your_project/
├─ alfworld_env.py       # 加载任务、执行动作、返回环境结果
├─ prepare_data.py       # 下载、导入、检查数据
├─ agent.py              # ReAct 循环、历史、ALFWorld 提示词；新方法改这里
├─ model.py              # 推理后端选择、Transformers 兼容实现
├─ vllm_model.py         # 本地 vLLM 加载、批量生成、token 统计
├─ hiper_react.py        # HiPER 普通 ReAct 提示词、历史与动作处理
├─ evaluate.py           # 逐任务评测、JSONL 日志、结果汇总
├─ configs/alfworld.yaml
├─ data/alfworld/
├─ models/Qwen2.5-1.5B-Instruct/
├─ requirements.txt
└─ LICENSE
```

运行时不依赖旁边的 PoS 或 AdaMEM 目录。
已准备完整 ALFWorld 文本任务数据：训练集 3553、seen 评测集 140、unseen 评测集 134，位于 `data/alfworld/`。
运行推荐 Linux / WSL、Python 3.11，
所需依赖记录在 requirements.txt。

## 从 PoS 保留了什么

- 环境类提取自 `_ALFWorld`，只改类名和数据路径。
- 保留 ALFWorld Raw/ReAct 提示词、任务示例、完整历史、JSON 修复和缺失动作重试。
- 保留 JSONL 日志格式；支持本地 vLLM / Transformers 推理，并统计实际 token。
- 将相关逻辑合并到上述文件，去掉其他 benchmark 提示词、belief 分支、上下文协议和 Markdown 渲染工具。
- 评测入口是独立适配版，没有 PoS 主程序的 worker 重启和续跑调度。
- 数据准备基于 PoS 下载/复制逻辑，增加划分选择和文件检查。

环境的成功标记来自 `won`；默认每个任务最多 50 个策略轮次。
有动作则交给环境执行，不在入口额外拦截不可用动作；缺失动作且修复失败也消耗轮次。
训练集的采样顺序遵循上游环境，不保证无重复遍历。
派生代码保留 PoS MIT LICENSE；环境和数据遵循各自上游许可。

## 使用

默认 vLLM 路径使用远程 Linux CUDA GPU 环境，无需账号、API key 或单独启动 HTTP 服务。
在独立环境中安装与远程 GPU/驱动匹配的 vLLM，再安装环境和评测依赖；不要叠加下文
Transformers 路径的旧 PyTorch/Transformers 版本约束。以下命令仅为远程运行说明：

```bash
conda create -n alfworld-vllm python=3.11 -y
conda activate alfworld-vllm
python -m pip install 'cmake>=3.18,<4'
python -m pip install -r requirements-vllm.txt
python evaluate.py --config configs/alfworld.yaml
# 全量 seen + unseen：
python evaluate.py --config configs/alfworld.yaml --split both
# 三卡独立副本，每卡最多推进 4 个任务：
python parallel_evaluate.py --config configs/alfworld.yaml --gpus 0,1,2 --split both --batch-size 4
```

`inference_backend: vllm` 在进程内加载本地权重，保留原来的单次调用、批量调用与 token
统计接口。默认 tensor_parallel_size=1、GPU 内存使用比例 0.6，输入/输出预算 2048/512。
使用 `generation_config="vllm"` 避免模型文件隐式覆盖采样参数，并显式传入配置中的参数与 EOS。
输出 token 数使用 vLLM 返回的 token_ids 统计，输入不计 padding。

`--gpus 0,1,2` 表示三个独立 worker，每个 worker 只看一张卡，不能同时配置 TP>1。
如需一个模型跨多卡，设置 `vllm.tensor_parallel_size` 并直接运行 `evaluate.py`，
通过 `CUDA_VISIBLE_DEVICES` 指定可见 GPU。环境批量大小仍由 `batch_size` 控制。
首次建立引擎可能有编译开销；切换推理引擎或批量大小不保证逐 token 结果一致。
依赖文件保留 vLLM 版本选择空间；正式实验需在远程环境固定并记录实际版本与硬件。

### 可选 Transformers 后端

推荐 Linux / WSL、Python 3.11。从本目录创建独立环境并安装依赖：

```bash
conda create -n alfworld-local python=3.11 -y
conda activate alfworld-local
# 先安装 CMake 3.x，供 TextWorld 的规划器源码构建使用。
python -m pip install 'cmake>=3.18,<4'
# RTX PRO 6000 Blackwell 使用支持该架构的 PyTorch CUDA 12.8 构建。
python -m pip install 'torch==2.11.0+cu128' --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
python -m pip install -r requirements.txt
```

`requirements.txt` 包含 ALFWorld/TextWorld、PyYAML、NumPy，以及本地推理使用的
PyTorch、Transformers 和 safetensors。TextWorld 的规划器需要 CMake 3.x、
GCC/G++ 和 make；安装时应先激活新环境，让构建脚本使用其中的 Python。
使用 GPU 时还需可用的 NVIDIA 驱动及匹配的 PyTorch CUDA 版本；
驱动显示的 CUDA 13.0 是其支持的版本上限，可以运行 CUDA 12.8 构建，安装上述
PyTorch 二进制包无需另装 CUDA Toolkit。

数据已经准备好，运行前可直接检查，无需重新下载：

```bash
python prepare_data.py --verify-only
# 如需重新准备数据或导入已有数据根目录（包含 json_2.1.1/valid_unseen/ 和 logic/）：
# python prepare_data.py
# python prepare_data.py --source /path/to/alfworld-data
```

默认准备 134 个 valid-unseen 任务；其他划分可传 `--split train` 或
`--split eval_in_distribution`。检查仅验证布局，不验证文件校验和或实际可解性。

配置里的 `model` 指向本地 `models/Qwen2.5-1.5B-Instruct`，相对路径以本项目目录为基准。
加载模型和 tokenizer 时仅读取本地文件。`device: auto` 优先使用 CUDA，
不可用时使用 CPU；也可指定 `cpu` 或 `cuda:0`。CPU 使用 float32，运行较慢，
仅模型权重约需 6.2 GB 内存，另需为上下文和运行时留出空间；CUDA 使用支持的
bfloat16 或 float16。默认先跑一例，`temperature: 0` 使用贪心生成。

```bash
python evaluate.py --config configs/alfworld_raw_transformers.yaml
```

`end_case: 134` 跑全量 unseen；范围为 `[start_case, end_case)`；null 表示到末尾。
数据和输出相对路径以本项目目录为基准。
模型上下文上限为 32768 tokens，输入加 `max_tokens` 超出上限时会报错，完整历史不自动截断。

## HiPER 普通 ReAct 基线

新增 `configs/alfworld_hiper_react.yaml`，将 HiPER 仓库的普通 ReAct 接口接入本框架。
配置 `react_style: hiper` 由 `create_agent()` 选择 `hiper_react.py` 的策略。
`configs/alfworld.yaml` 与该命名配置均默认使用 vLLM；原完整历史 JSON ReAct
可通过 `configs/alfworld_raw_transformers.yaml` 选择。

- 使用上游原始 ALFWorld 提示词，每轮传入一个 user 消息，不额外添加本项目的 system 提示词或示例。
- 输出格式为 `<think>...</think><action>...</action>`，每轮生成一次并执行一次环境步骤。
- 默认只输入最近 2 步的“执行前观测 + 已执行动作”；不向后续提示词回填 thought。
  完整交互轨迹和 thought 仍保存在日志和 result.json 中。
- 动作转小写，提示词排除 `help`；不额外检查动作是否属于可用动作列表。
  缺少动作标签时执行小写输出的最后 30 个字符；格式不合规的输出照常消耗环境步骤，
  不进行 JSON 修复或缺失动作重试。
- `is_action_valid` 沿用上游的格式标记，检查 action/think 标签及中文字符，
  不代表动作一定能执行成功；该字段保存在决策日志和 trajectory 中，不改变环境成功判定。
- 新配置采用上游普通 ReAct 脚本的验证采样设置：temperature=0.4、top_p=1、禁用 top-k、
  repetition_penalty=1；最多 2048 输入 tokens、512 输出 tokens、50 步。
  超出输入预算直接报错。`seed: 0` 设置模型采样与环境种子；批大小、推理引擎与设备仍会影响结果。

先跑一个 unseen 任务、全量评测、三卡评测分别使用：

```bash
python evaluate.py --config configs/alfworld_hiper_react.yaml
python evaluate.py --config configs/alfworld_hiper_react.yaml --split both
python parallel_evaluate.py --config configs/alfworld_hiper_react.yaml --gpus 0,1,2 --split both --batch-size 4
```

这里使用本地 Qwen 权重和 vLLM 推理，移植的是普通 ReAct 的提示词、历史与动作处理，
不包含 HiPER 的 Plan-Execute、HAE 或强化学习训练。该配置不是论文训练后成绩的完整复现；
与原全历史 ReAct 做严格消融时，还应对齐采样参数、输入预算、模型权重及测试任务。
许可证与来源见 `THIRD_PARTY_NOTICES.md`。

回归用例位于 `tests/test_hiper_react.py`，覆盖提示词、历史对齐、无效输出、原策略及串行/批量接入。
依赖安装后的验证命令为 `python -m unittest discover -s tests -v`。

## 开发新方法

从 `agent.py` 的 `Agent` 类开始；修改 `_build_messages()` 可改变模型输入，
修改 `run()` 可改变规划、记忆和交互流程。默认 raw 历史辅助类也在同一文件。
每个任务新建 Agent，constructor 接收 `(config, call_model, logger)`。

固定环境接口：

```python
state = env.reset()
state = env.step(action)
# case_id / task / observation / actions / done / success / score
```

Agent.run(env) 返回 case_id、task、success、score、steps、trajectory。
评测保存到 `outputs/<时间戳>/cases/case_0000/` 的 events.jsonl 和 result.json，
并生成 summary.json。每次运行独立建目录；基础设施异常保存 error.json 后中止，
不计成普通任务失败。summary 只统计已完成任务。
token usage 来自实际输入和生成的 token 数，输出计数包含 EOS 等特殊 token，
JSON 修复产生的额外模型调用也计入统计。


## 全量评测与六类任务成功率

```bash
conda activate alfworld-local
python evaluate.py --config configs/alfworld.yaml --split both
# 单独评测：--split seen 或 --split unseen
```

`--split` 覆盖 YAML 的任务范围和类型，运行全部六类任务。seen 为 140 个，unseen 为 134 个。
两个划分共享模型，环境和结果目录分别独立。不传此参数仍按原配置运行。
每个任务后的 summary.json 增加 by_task_type；每个划分完成或异常退出时生成 charts/：
seen_success_rates、unseen_success_rates、seen_unseen_comparison，每张提供 PNG/SVG。
task_summary.json 记录每类可评测数、已完成数、成功数和成功率。
成功率以已完成任务为分母，未评测类别显示 N/A（JSON 为 null），不能当成 0%。
覆盖数量采用底层环境相同的六类、非 Sliced、非 movable、solvable 游戏筛选规则。

已有结果可独立生成图表，无需模型推理：

```bash
python report.py --run-dir outputs/某次运行目录
```

指定一次运行的目录，不要混合重复任务；重复 case_id 会报错。
六类依次为拾取放置、灯光下查看、清洗放置、加热放置、冷却放置、拾取两个放置。
图中标注成功数/已完成数；部分结果的图不代表完整测试集成绩。

## 运行进度

评测默认显示两层 tqdm 进度条：外层是当前划分的已完成任务数、耗时、预计剩余时间、
成功数和成功率；内层是当前任务已消耗的策略轮数、实际动作数和当前阶段。
`generating (attempt N)` 表示模型正在生成或修复 JSON，`executing` 显示正在执行的动作。
模型生成过程中没有逐 token 进度；完成一轮后更新轮数。内层总数是轮数预算，
任务提前结束时不会补满。两个划分依次运行，各自显示任务进度。
每个任务结束后打印其标识、成功状态、动作数和耗时。

运行命令不变；重定向日志时可加 `--no-progress` 关闭动态进度条，保留任务完成信息。
这条流程是推理评测，进度不代表模型参数训练。

## 三张 GPU 并行评测

先确保运行环境能看到三张 GPU（`nvidia-smi`）。每张卡单独加载一份模型，任务按
排序后的编号取 `rank::3`，互不重复；每个进程先跑自己分到的 seen，再跑 unseen。

```bash
conda activate alfworld-local
cd /srv/lab-gpu/home/gpu_zzzz_b0e26b44/your_project
python parallel_evaluate.py --config configs/alfworld.yaml --gpus 0,1,2 --split both --batch-size 4
```

完整数据分配为 GPU 0：47 seen + 45 unseen，GPU 1：47 + 45，GPU 2：46 + 44。
`--gpus` 指当前进程可见 GPU 的编号；若已有 CUDA_VISIBLE_DEVICES 设置，会映射到
对应设备。设备不足或编号重复时会在加载模型前报错。

终端显示每张卡的任务进度、当前划分、轮次、动作数和阶段。详细日志在本次目录的
`worker_0/worker.log`、`worker_1/worker.log`、`worker_2/worker.log`。
每个 worker 下的 valid_seen/、valid_unseen/ 保留独立结果，任务目录编号仍为原始编号。
主进程结束时统一生成根目录 summary.json 和 charts/ 三张图，并检查重复任务与完成数量。
任一进程失败会停止其余进程，保留已完成的任务和部分报告；Ctrl+C 也会清理子进程。
不支持自动续跑。单独重画图可用 `python report.py --run-dir outputs/本次并行运行目录`。

每卡默认 `--batch-size 4`：最多同时推进 4 个独立任务，三卡合计最多 12 个。
每卡只加载一份模型，把当前各任务的请求左侧补齐后合批生成；任务结束后补入下一任务，
尾批可以小于 4。历史、环境、JSON 重试、token 统计均独立，原动作预算不变。
进度中的 batch 是当前批次任务数；turn/actions 为活跃任务中的最大值，不是单个任务的进度。
改用 `--batch-size 1` 可回到串行调度。批大小可能造成浮点差异，不能保证逐任务结果完全一致。
运行中的旧进程不会自动切换批大小，需重新启动；新运行写入新目录，不支持断点续跑。
