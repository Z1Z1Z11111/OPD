"""Offline vLLM inference for HiPER-style ReAct; no HTTP server or API key.

This adapter supplies generation, not an RL trainer or actor weight synchronization.
Heavy dependencies are imported only when the backend is selected.
"""
import json

from alfworld_env import project_path


def sampling_options(config: dict) -> dict:
    """Translate the evaluator's decoding settings to vLLM SamplingParams."""
    temperature = config["temperature"]
    max_tokens = config["max_tokens"]
    top_k = config.get("top_k", 0)
    top_p = config.get("top_p", 1.0)
    penalty = config.get("repetition_penalty", 1.0)
    if temperature < 0 or max_tokens <= 0:
        raise ValueError("temperature must be nonnegative and max_tokens must be positive")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < -1:
        raise ValueError("top_k must be -1, 0 (disabled), or a positive integer")
    if not 0 < top_p <= 1 or penalty <= 0:
        raise ValueError("top_p must be in (0, 1] and repetition_penalty must be positive")
    return dict(n=1, temperature=temperature, max_tokens=max_tokens,
                top_k=-1 if top_k <= 0 else top_k, top_p=top_p,
                repetition_penalty=penalty, detokenize=False)


class VLLMModelCaller:
    """Load one local policy per process and batch independent conversations."""

    def __init__(self, config: dict):
        self.config = config
        model_path = project_path(config["model"])
        if not model_path.is_dir():
            raise FileNotFoundError(f"Local model directory not found: {model_path}")
        options = sampling_options(config)
        device = config.get("device", "auto")
        if device not in ("auto", "cuda", "cuda:0"):
            raise ValueError("vLLM requires CUDA here; select GPUs with CUDA_VISIBLE_DEVICES")
        settings = config.get("vllm", {})
        allowed = {"tensor_parallel_size", "gpu_memory_utilization", "dtype", "max_model_len",
                   "max_num_seqs", "enforce_eager", "enable_prefix_caching", "enable_chunked_prefill"}
        unknown = set(settings) - allowed
        if unknown:
            raise ValueError(f"Unknown vllm settings: {sorted(unknown)}")
        tp = settings.get("tensor_parallel_size", 1)
        if isinstance(tp, bool) or not isinstance(tp, int) or tp < 1:
            raise ValueError("vllm.tensor_parallel_size must be a positive integer")
        utilization = settings.get("gpu_memory_utilization", 0.6)
        if not 0 < utilization <= 1:
            raise ValueError("vllm.gpu_memory_utilization must be in (0, 1]")
        self.prompt_limit = config.get("max_prompt_tokens", 2048)
        if not isinstance(self.prompt_limit, int) or self.prompt_limit < 1:
            raise ValueError("max_prompt_tokens must be a positive integer")
        self.context_limit = settings.get("max_model_len", self.prompt_limit + config["max_tokens"])
        if self.context_limit < self.prompt_limit + config["max_tokens"]:
            raise ValueError("vllm.max_model_len must cover max_prompt_tokens + max_tokens")

        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("The vLLM backend requires a CUDA GPU; use the remote Linux GPU environment")
        if tp > torch.cuda.device_count():
            raise ValueError("tensor_parallel_size exceeds the number of visible CUDA GPUs")
        try:
            from vllm import LLM, SamplingParams
        except ImportError as error:
            raise ImportError("Install requirements-vllm.txt in the remote GPU environment") from error

        self.engine = LLM(
            model=str(model_path), tokenizer=str(model_path), trust_remote_code=False,
            generation_config="vllm", seed=int(config.get("seed", 0)),
            tensor_parallel_size=tp, gpu_memory_utilization=utilization,
            dtype=settings.get("dtype", "auto"), max_model_len=self.context_limit,
            max_num_seqs=settings.get("max_num_seqs", 32),
            enforce_eager=settings.get("enforce_eager", False),
            enable_prefix_caching=settings.get("enable_prefix_caching", True),
            enable_chunked_prefill=settings.get("enable_chunked_prefill", False),
        )
        self.tokenizer = self.engine.get_tokenizer()
        # Match the local checkpoint's EOS set while overriding its sampling defaults.
        generation_file = model_path / "generation_config.json"
        if generation_file.is_file():
            eos = json.loads(generation_file.read_text(encoding="utf-8")).get("eos_token_id")
            if eos is not None:
                options["stop_token_ids"] = eos if isinstance(eos, list) else [eos]
        self.sampling_params = SamplingParams(**options)
        self.usage = dict(input_tokens=0, output_tokens=0, total_tokens=0)

    def __call__(self, messages):
        return self.generate_batch([messages])[0][0]

    def generate_batch(self, conversations):
        if not conversations:
            return []
        requests = []
        for messages in conversations:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            if len(ids) > self.prompt_limit:
                raise ValueError(f"Prompt ({len(ids)} tokens) exceeds max_prompt_tokens ({self.prompt_limit})")
            if len(ids) + self.config["max_tokens"] > self.context_limit:
                raise ValueError("Prompt plus output budget exceeds vLLM context limit")
            requests.append({"prompt_token_ids": ids})
        outputs = self.engine.generate(requests, self.sampling_params, use_tqdm=False)
        if len(outputs) != len(requests):
            raise RuntimeError("vLLM returned a different number of batch responses")
        results = []
        for request, output in zip(requests, outputs):
            if output.prompt_token_ids != request["prompt_token_ids"] or len(output.outputs) != 1:
                raise RuntimeError("Unexpected vLLM response ordering or number of completions")
            completion_ids = list(output.outputs[0].token_ids)
            prompt_tokens = len(request["prompt_token_ids"])
            usage = dict(input_tokens=prompt_tokens, output_tokens=len(completion_ids),
                         total_tokens=prompt_tokens + len(completion_ids))
            for key in self.usage:
                self.usage[key] += usage[key]
            results.append((self.tokenizer.decode(completion_ids, skip_special_tokens=True), usage))
        return results

    def token_usage(self):
        return self.usage.copy()
