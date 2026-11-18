"""Run the local Qwen checkpoint and count prompt/completion tokens."""

from alfworld_env import project_path


def generation_kwargs(config: dict, pad_token_id: int) -> dict:
    """Build explicit overrides without changing legacy checkpoint defaults."""
    temperature = config["temperature"]
    options = dict(max_new_tokens=config["max_tokens"],
                   do_sample=temperature > 0, pad_token_id=pad_token_id)
    if temperature > 0:
        options["temperature"] = temperature
        for name in ("top_p", "top_k"):
            if name in config:
                options[name] = config[name]
    else:
        # Reset the checkpoint's sampling defaults for greedy decoding.
        options.update(temperature=1.0, top_p=1.0, top_k=50)
    if "repetition_penalty" in config:
        options["repetition_penalty"] = config["repetition_penalty"]
    return options


class ModelCaller:
    """Keep the evaluator API stable while selecting a local inference backend."""

    def __init__(self, config: dict):
        backend = config.get("inference_backend", "transformers")
        if backend == "vllm":
            from vllm_model import VLLMModelCaller
            self.backend = VLLMModelCaller(config)
        elif backend == "transformers":
            self.backend = TransformersModelCaller(config)
        else:
            raise ValueError(f"Unsupported inference_backend: {backend}")

    def __call__(self, messages):
        return self.backend(messages)

    def generate_batch(self, conversations):
        return self.backend.generate_batch(conversations)

    def token_usage(self):
        return self.backend.token_usage()


class TransformersModelCaller:
    """Load a local Transformers model once for all evaluation episodes."""

    def __init__(self, config: dict):
        self.config = config
        model_path = project_path(config["model"])
        if not model_path.is_dir():
            raise FileNotFoundError(f"Local model directory not found: {model_path}")
        if config["temperature"] < 0:
            raise ValueError("temperature must be nonnegative")
        if config["max_tokens"] <= 0:
            raise ValueError("max_tokens must be positive")
        if config.get("max_prompt_tokens", 1) <= 0:
            raise ValueError("max_prompt_tokens must be positive")
        if "top_p" in config and not 0 < config["top_p"] <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if "top_k" in config and (not isinstance(config["top_k"], int) or config["top_k"] < 0):
            raise ValueError("Transformers top_k must be a nonnegative integer")
        if config.get("repetition_penalty", 1.0) <= 0:
            raise ValueError("repetition_penalty must be positive")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        device = config.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; set device to cpu or fix the GPU driver/PyTorch installation")
            with torch.cuda.device(self.device):
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif self.device.type == "cpu":
            dtype = torch.float32
        else:
            raise ValueError("device must be auto, cpu, or cuda (optionally cuda:<index>)")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path), local_files_only=True,
            use_safetensors=True, dtype=dtype,
        ).to(self.device).eval()
        if config.get("seed") is not None:
            torch.manual_seed(int(config["seed"]))
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0

    def __call__(self, messages: list[dict]) -> str:
        return self.generate_batch([messages])[0][0]

    def generate_batch(self, conversations):
        """Return (response, token usage) for each independent conversation."""
        if not conversations:
            return []
        prompts = [self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        ) for messages in conversations]
        inputs = self.tokenizer(
            prompts, return_tensors="pt", add_special_tokens=False, padding=True,
        ).to(self.device)
        input_length = inputs["input_ids"].shape[-1]
        prompt_limit = self.config.get("max_prompt_tokens")
        if prompt_limit is not None and input_length > prompt_limit:
            raise ValueError(
                f"Prompt ({input_length} tokens) exceeds max_prompt_tokens ({prompt_limit})"
            )
        max_tokens = self.config["max_tokens"]
        context_limit = self.model.config.max_position_embeddings
        if input_length + max_tokens > context_limit:
            raise ValueError(
                f"Prompt ({input_length} tokens) plus max_tokens ({max_tokens}) "
                f"exceeds the model context limit ({context_limit})"
            )

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        generation = generation_kwargs(self.config, pad_token_id)
        with self.torch.inference_mode():
            outputs = self.model.generate(**inputs, **generation)
        eos = self.model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
        results = []
        for row, mask in zip(outputs, inputs["attention_mask"]):
            completion = row[input_length:].tolist()
            # Include the first EOS, exclude padding inserted after termination.
            stop = next((i + 1 for i, token in enumerate(completion) if token in eos_ids), len(completion))
            completion = completion[:stop]
            prompt_tokens = int(mask.sum().item())
            usage = dict(input_tokens=prompt_tokens, output_tokens=len(completion),
                         total_tokens=prompt_tokens + len(completion))
            self.input_tokens += usage["input_tokens"]
            self.output_tokens += usage["output_tokens"]
            results.append((self.tokenizer.decode(completion, skip_special_tokens=True), usage))
        self.total_tokens = self.input_tokens + self.output_tokens
        return results

    def token_usage(self) -> dict[str, int]:
        """Return cumulative local token counts, including generated EOS tokens."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }
