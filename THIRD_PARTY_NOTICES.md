# Third-party code

The existing PoS-derived code remains covered by the root MIT LICENSE.

`hiper_react.py` adapts the ordinary ALFWorld ReAct interface distributed in
HiPER-agent (the local `HiPER-agent-main` source tree), which builds on
verl-agent / GiGPO:

- `agent_system/environments/prompts/alfworld.py`: ordinary ReAct templates,
  preserved verbatim.
- `agent_system/environments/env_package/alfworld/projection.py`: action
  projection behavior, adapted to the local decision/logging interface.
- `agent_system/memory/memory.py` and `agent_system/environments/env_manager.py`:
  recent pre-action observation/action history and prompt assembly, adapted for
  one episode at a time.

These portions are licensed under Apache License 2.0; see
`licenses/Apache-2.0.txt`. The original copyright header is retained in
`hiper_react.py`. Local changes replace batched Ray environment management with
the existing episode generator and export format.

The upstream Notice.txt records:

> This project includes code from the verl project.
>
> Copyright 2023-2024 Bytedance Ltd. and/or its affiliates
>
> Additional modifications and contributions:
>
> Copyright 2025 Nanyang Technological University (NTU), Singapore and the verl-agent (GiGPO) team.
>
> Copyright 2026 University of Minnesota, OptimAI-Lab.

The ALFWorld GRPO training sources in `rl/hiper/` come from the local HiPER-agent checkout based on commit `ec5982a71635242de983a34407def05ac0d2e349`, including its HF/SDPA worker adaptations. Upstream notices and license are included in `rl/hiper/Notice.txt` and `rl/hiper/LICENSE`.
