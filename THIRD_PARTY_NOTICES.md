# Third-party code

The PoS-derived evaluation runner, environment adapter, and data preparation
code remain covered by the root MIT [LICENSE](LICENSE).

`agent.py` adapts the ordinary ALFWorld ReAct prompt templates, recent-window
history, and action projection from HiPER-agent (`git@github.com:JonP07/HiPER-agent.git`),
which builds on verl-agent / verl. These portions are licensed under Apache
License 2.0 (see [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt)).

The minimal GRPO training subset in `rl/hiper/` is pruned from the same
HiPER-agent checkout (`ec5982a71635242de983a34407def05ac0d2e349`), keeping only
the ALFWorld ReAct environment wrapper, multi-turn rollout loop, episode reward
manager, and FSDP + vLLM/HF GRPO trainer. Plan–Execute (`Options`), HAE,
GiGPO, other benchmark environments, Megatron, SGLang, and auxiliary trainers
have been removed. Upstream notices and license are retained in
`rl/hiper/Notice.txt` and `rl/hiper/LICENSE`.

The upstream `Notice.txt` records:

> This project includes code from the verl project.
>
> Copyright 2023-2024 Bytedance Ltd. and/or its affiliates
>
> Additional modifications and contributions:
>
> Copyright 2025 Nanyang Technological University (NTU), Singapore and the verl-agent (GiGPO) team.
>
> Copyright 2026 University of Minnesota, OptimAI-Lab.
