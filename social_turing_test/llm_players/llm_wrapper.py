import json
import os
import random
import time
from contextlib import nullcontext
from functools import cache
import requests
from google import genai
from google.genai import types as genai_types

from social_turing_test.core.game_constants import get_current_timestamp
from social_turing_test.llm_players.llm_constants import (TASK2OUTPUT_FORMAT, INITIAL_GENERATION_PROMPT, \
                                       INSTRUCTION_INPUT_RESPONSE_PATTERN, LLAMA3_PATTERN, DEFAULT_PROMPT_PATTERN,
                                       NUM_BEAMS_KEY, \
                                       MODEL_NAME_KEY, USE_PIPELINE_KEY, PIPELINE_TASK_KEY, MAX_NEW_TOKENS_KEY,
                                       GENERAL_SYSTEM_INFO, \
                                       REPETITION_PENALTY_KEY, USE_TOGETHER_KEY, TOGETHER_API_KEY_KEYWORD, \
                                       SLEEPING_TIME_FOR_API_GENERATION_ERROR, USE_GEMINI_KEY,
                                       GEMINI_API_KEY_KEYWORD, GEMINI_GENERATION_PARAMETERS,
                                       TOGETHER_GENERATION_PARAMETERS, HUGGINGFACE_GENERATION_PARAMETERS,
                                       TOGETHER_SCHEDULING_GENERATION_PARAMETERS,
                                       HUGGINGFACE_SCHEDULING_GENERATION_PARAMETERS,
                                       GEMINI_SCHEDULING_GENERATION_PARAMETERS, QWEN_PATTERN,
                                       TEMPERATURE_KEY, MAX_TOKENS_KEY, USE_OPENROUTER_KEY,
                                       OPENROUTER_GENERATION_PARAMETERS,
                                       OPENROUTER_SCHEDULING_GENERATION_PARAMETERS)
from social_turing_test.llm_players.openrouter_utils import (model_supports_temperature, is_unsupported_param_error,
                                          min_max_tokens_for, is_max_tokens_too_low_error,
                                          reasoning_payload_for)

# torch + transformers are ONLY needed for the local HuggingFace pipeline path
# (use_pipeline / local model). Cloud providers (Gemini/Vertex, Together,
# OpenRouter) don't need them. Importing torch is slow and can thrash a small
# server's disk for minutes, so we DEFER the import until a local-model path
# actually asks for it instead of importing it at module load.
torch = None
AutoTokenizer = AutoModelForCausalLM = AutoModelForSeq2SeqLM = AutoConfig = pipeline = None
_hf_imports_loaded = False


def _ensure_hf_imports():
    """Import torch + transformers on first local-model use; no-op for cloud providers.

    Returns True if torch/transformers are available, False otherwise.
    """
    global torch, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
    global AutoConfig, pipeline, _hf_imports_loaded
    if _hf_imports_loaded:
        return torch is not None
    _hf_imports_loaded = True
    try:
        print("Trying to import torch...", get_current_timestamp())
        import torch as _torch
        print("Finished importing torch!", get_current_timestamp())
        print("Trying to import from transformers...", get_current_timestamp())
        from transformers import (AutoTokenizer as _AutoTokenizer,
                                  AutoModelForCausalLM as _AutoModelForCausalLM,
                                  AutoModelForSeq2SeqLM as _AutoModelForSeq2SeqLM,
                                  AutoConfig as _AutoConfig, pipeline as _pipeline)
        print("Finished importing from transformers!", get_current_timestamp())
        torch = _torch
        AutoTokenizer, AutoModelForCausalLM = _AutoTokenizer, _AutoModelForCausalLM
        AutoModelForSeq2SeqLM, AutoConfig, pipeline = _AutoModelForSeq2SeqLM, _AutoConfig, _pipeline
        return True
    except ModuleNotFoundError as _hf_exc:
        print(f"torch/transformers unavailable ({_hf_exc}); local HuggingFace pipeline "
              f"path disabled - cloud providers (Gemini/Together/OpenRouter) still work.",
              flush=True)
        return False

# The Together SDK is only needed for the Together provider path; import it
# optionally so Gemini/Vertex and OpenRouter games run without it installed.
try:
    from together import Together
    from together.error import TogetherException
except ModuleNotFoundError:
    Together = None
    class TogetherException(Exception):  # type: ignore
        """Placeholder so `except TogetherException` is valid when the SDK is absent."""

CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")


def is_local_path(model_name):
    return os.path.isdir(model_name)  # maybe should come up with better mechanism


@cache
def cached_model(model_name):
    _ensure_hf_imports()
    if is_local_path(model_name):
        config = AutoConfig.from_pretrained(model_name)
        return AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config)
    return AutoModelForCausalLM.from_pretrained(model_name, cache_dir=CACHE_DIR)


@cache
def cached_tokenizer(model_name):
    _ensure_hf_imports()
    return AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)


@cache
def cached_pipeline(model_name, task):
    _ensure_hf_imports()
    return pipeline(task, model_name, device_map="auto")


def get_api_key(key_type):
    """Read only the operator's environment; never load private key files."""
    return os.environ.get(key_type) or None

class LLMWrapper:

    # Per-model $/1M token pricing (input, output). Update as pricing changes.
    _PRICE_PER_M = {
        "gemini-2.5-pro":        (1.25, 10.0),
        "gemini-2.5-flash":      (0.15,  0.60),
        "gemini-2.5-flash-lite": (0.10,  0.40),
        "gemini-2.0-flash":      (0.10,  0.40),
    }
    _DEFAULT_PRICE = (0.5, 1.5)

    @classmethod
    def _estimate_cost(cls, model_name: str, input_tokens: int, output_tokens: int) -> float:
        key = next((k for k in cls._PRICE_PER_M if model_name.startswith(k)), None)
        pin, pout = cls._PRICE_PER_M.get(key, cls._DEFAULT_PRICE)
        return round((input_tokens * pin + output_tokens * pout) / 1_000_000, 6)

    def __init__(self, logger, **llm_config):
        self.logger = logger
        self.model_name = llm_config[MODEL_NAME_KEY]
        self._last_usage: dict | None = None
        self.use_together = llm_config.get(USE_TOGETHER_KEY, False)
        self.use_pipeline = llm_config.get(USE_PIPELINE_KEY, False)
        self.use_gemini = llm_config.get(USE_GEMINI_KEY, False)
        self.use_openrouter = llm_config.get(USE_OPENROUTER_KEY, False)
        self.pipeline_task = llm_config.get(PIPELINE_TASK_KEY, "text-generation")
        # Default to CPU; the local-model branch below imports torch and sets a
        # real torch.device. Cloud providers never touch torch, so it stays "cpu".
        self.device = "cpu"
        self.prompt_template = self._get_prompt_template()

        if self.use_gemini:
            api_key = get_api_key(GEMINI_API_KEY_KEYWORD) or get_api_key("GOOGLE_API_KEY")
            if api_key:
                print(f"[Gemini auth] Using API key for model {self.model_name}", flush=True)
                self.client = genai.Client(api_key=api_key)
            else:
                project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
                location = os.environ.get("GOOGLE_CLOUD_LOCATION")
                if not project_id or not location:
                    raise RuntimeError(
                        "Gemini authentication missing. Set GEMINI_API_KEY or GOOGLE_API_KEY, "
                        "or explicitly set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION "
                        "for your own Vertex AI project with Application Default Credentials."
                    )
                print(f"[Gemini auth] Using Vertex AI (project={project_id}, location={location}) for model {self.model_name}", flush=True)
                self.client = genai.Client(vertexai=True, project=project_id, location=location)
            self.pipeline = self.tokenizer = self.model = None
            generation_params = GEMINI_GENERATION_PARAMETERS
            scheduling_generation_params = GEMINI_SCHEDULING_GENERATION_PARAMETERS
        elif self.use_openrouter:
            self.openrouter_api_key = get_api_key("OPENROUTER_API_KEY")
            if not self.openrouter_api_key:
                raise RuntimeError(
                    "OpenRouter API key missing. Set OPENROUTER_API_KEY env var."
                )
            self.client = self.pipeline = self.tokenizer = self.model = None
            generation_params = OPENROUTER_GENERATION_PARAMETERS
            scheduling_generation_params = OPENROUTER_SCHEDULING_GENERATION_PARAMETERS
        elif self.use_together:
            api_key = get_api_key(TOGETHER_API_KEY_KEYWORD)
            if not api_key:
                raise RuntimeError("Together API key missing. Set TOGETHER_API_KEY env var.")
            if Together is None:
                raise RuntimeError("Together SDK missing. Install the together package.")
            self.client = Together(api_key=api_key)
            self.pipeline = self.tokenizer = self.model = None
            generation_params = TOGETHER_GENERATION_PARAMETERS
            scheduling_generation_params = TOGETHER_SCHEDULING_GENERATION_PARAMETERS
        elif self.use_pipeline:
            _ensure_hf_imports()
            self.pipeline = cached_pipeline(self.model_name, self.pipeline_task)
            self.client = self.tokenizer = self.model = None
            generation_params = HUGGINGFACE_GENERATION_PARAMETERS
            scheduling_generation_params = HUGGINGFACE_SCHEDULING_GENERATION_PARAMETERS
        else:
            raise ValueError(
                "No model provider selected. Set use_gemini, use_openrouter, or "
                "use_together; local HuggingFace requires explicit use_pipeline=True."
            )

        self.generation_parameters = {key: value for key, value in llm_config.items()
                                      if key in generation_params}
        
        # For scheduling, use hardcoded defaults - don't let config override
        # This ensures scheduling always uses minimal tokens (just for <send>/<wait>)
        if self.use_gemini:
            self.scheduling_generation_parameters = GEMINI_SCHEDULING_GENERATION_PARAMETERS.copy()
        elif self.use_openrouter:
            self.scheduling_generation_parameters = OPENROUTER_SCHEDULING_GENERATION_PARAMETERS.copy()
        elif self.use_together:
            self.scheduling_generation_parameters = TOGETHER_SCHEDULING_GENERATION_PARAMETERS.copy()
        else:
            self.scheduling_generation_parameters = HUGGINGFACE_SCHEDULING_GENERATION_PARAMETERS.copy()

        if (NUM_BEAMS_KEY in self.generation_parameters
            and self.generation_parameters[NUM_BEAMS_KEY] < 2):
            del self.generation_parameters[NUM_BEAMS_KEY]

        # initial generation just to save time of first generation in real time
        self.generate(INITIAL_GENERATION_PROMPT, False, system_info=GENERAL_SYSTEM_INFO)

    def _get_prompt_template(self):
        model_name = self.model_name.lower()
        if "phi-3" in model_name:
            return INSTRUCTION_INPUT_RESPONSE_PATTERN
        elif "llama-3" in model_name:
            return LLAMA3_PATTERN
        # elif "____" in model_name: return "____"
        elif "qwen" in model_name:
            return QWEN_PATTERN
        else:
            return DEFAULT_PROMPT_PATTERN

    def pipeline_preprocessing(self, input_text, system_info):
        if self.use_together or self.use_openrouter or self.prompt_template in (INSTRUCTION_INPUT_RESPONSE_PATTERN, LLAMA3_PATTERN, QWEN_PATTERN):
            system_message = [{"role": "system", "content": system_info}] if system_info else []
            return system_message + [{"role": "user", "content": input_text}]
        else:
            raise NotImplementedError("Used model doesn't support pipeline, "
                                      "try `use_pipeline=False` in config")

    def direct_preprocessing(self, input_text, system_info) -> str:
        if self.prompt_template == INSTRUCTION_INPUT_RESPONSE_PATTERN:
            instruction = system_info.strip() + " " + input_text if system_info else input_text
            return f"### Instruction:\n{instruction}\n### Response: "
        # elif self.prompt_template is of Phi-3 style:
            # return f"<|user|>{input_text}<|end|>\n<|assistant|>"
        elif self.prompt_template == LLAMA3_PATTERN:
            if system_info:
                system_prompt = f"<|start_header_id|>system<|end_header_id|>{system_info}<|eot_id|>"
            else:
                system_prompt = ""
            return f"<|begin_of_text|>{system_prompt}" \
                   f"<|start_header_id|>user<|end_header_id|>{input_text}<|eot_id|>" \
                   f"<|start_header_id|>assistant<|end_header_id|>"
        else:
            raise NotImplementedError("Missing prompt template for used model")

    def direct_postprocessing(self, decoded_output):
        if self.prompt_template == INSTRUCTION_INPUT_RESPONSE_PATTERN:
            output = decoded_output.split("### Response:")[1].strip().split("</s>")[0]

            return output.strip()
        elif self.prompt_template == LLAMA3_PATTERN:
            assistant_prefix = "<|start_header_id|>assistant<|end_header_id|>"
            if assistant_prefix in decoded_output:
                decoded_output = decoded_output.split(assistant_prefix)[1].strip()
            decoded_output = decoded_output.split("<|eot_id|>")[0].split("<|end_of_text|>")[0].strip()
            if decoded_output and decoded_output[0] == ":":
                decoded_output = decoded_output[1:]
            if "<|eom_id|>" in decoded_output:
                decoded_output = decoded_output.split("<|eom_id|>")[0].strip()
            return decoded_output
        else:
            raise NotImplementedError("Missing output template for used model")

    def pop_usage(self) -> dict | None:
        """Return and clear the last recorded usage info, or None if unavailable."""
        u = self._last_usage
        self._last_usage = None
        return u

    def generate(self, input_text, is_scheduler=False, system_info="", max_output_tokens=None, component=None):
        if is_scheduler:
            generation_parameters = self.scheduling_generation_parameters
        else:
            generation_parameters = self.generation_parameters
        # Per-call output-length override. The chat generator runs with a tiny
        # max_tokens (e.g. 25) so messages stay 2-6 words; but background
        # sub-agents that emit structured JSON (Social State Tracker, Internet Context,
        # Persona Monitor) need far more room or their output is truncated
        # mid-object and fails to parse. Setting both token keys covers provider
        # paths that use either max_tokens or max_new_tokens.
        if max_output_tokens is not None:
            generation_parameters = dict(generation_parameters)
            generation_parameters[MAX_TOKENS_KEY] = max_output_tokens
            generation_parameters[MAX_NEW_TOKENS_KEY] = max_output_tokens
        # Which component (generator / scheduler / a sub-agent / reviewer) made
        # this call - used to route the log into a focused per-component file.
        comp = component or ("scheduler" if is_scheduler else "generator")
        self._last_usage = None
        call_start = time.perf_counter()
        # torch.inference_mode() only matters for the local HF tensor path; for
        # cloud providers it's a harmless no-op, so skip it when torch is absent.
        with (torch.inference_mode() if torch is not None else nullcontext()):
            if self.use_gemini:
                final_output = self.generate_with_gemini(input_text, system_info, generation_parameters)
            elif self.use_openrouter:
                messages = self.pipeline_preprocessing(input_text, system_info)
                final_output = self.generate_with_openrouter_safely(messages, generation_parameters)
            elif self.use_together:
                messages = self.pipeline_preprocessing(input_text, system_info)
                final_output = self.generate_with_together_safely(messages, generation_parameters)  # max_new_tokens -> max_tokens
            elif self.use_pipeline:
                messages = self.pipeline_preprocessing(input_text, system_info)
                outputs = self.pipeline(messages, **generation_parameters)
                final_output = outputs[0][TASK2OUTPUT_FORMAT[self.pipeline_task]][-1]
            else:
                prompt = self.direct_preprocessing(input_text, system_info)
                inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                outputs = self.model.generate(**inputs, **generation_parameters)
                decoded_output = self.tokenizer.decode(outputs[0])
                final_output = self.direct_postprocessing(decoded_output)
        # One grouped, readable entry per call: system prompt + user prompt +
        # output together, in both the per-component file and the master log.
        latency_s = time.perf_counter() - call_start
        meta = {}
        usage = self._last_usage or {}
        if usage:
            meta["tokens"] = f"{usage.get('input_tokens', '?')}->{usage.get('output_tokens', '?')}"
        cost = usage.get("estimated_cost_usd")
        self.logger.log_call(
            comp, system_info, input_text, final_output,
            latency_s=latency_s, cost=cost, meta=meta or None,
        )
        return final_output.replace("\n", "   ").strip()

    def generate_with_together_safely(self, messages, generation_parameters):
        MAX_RETRIES = 10
        output = None
        attempts = 0
        last_error = None
        while not output:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    **generation_parameters
                )
                output = response.choices[0].message.content
                if hasattr(response, 'usage') and response.usage:
                    u = response.usage
                    it, ot = getattr(u, 'prompt_tokens', 0), getattr(u, 'completion_tokens', 0)
                    self._last_usage = {"input_tokens": it, "output_tokens": ot,
                                        "estimated_cost_usd": self._estimate_cost(self.model_name, it, ot)}
            except TogetherException as e:
                attempts += 1
                last_error = e
                print(f"TogetherException ({attempts}/{MAX_RETRIES})\n{e}")
                self.logger.log("error generating with TogetherAI", f"Attempt {attempts}/{MAX_RETRIES}: {e}")
                if attempts >= MAX_RETRIES:
                    raise RuntimeError(
                        f"Together AI model '{self.model_name}' failed after {MAX_RETRIES} attempts: {last_error}"
                    )
                time.sleep(SLEEPING_TIME_FOR_API_GENERATION_ERROR)
        return output

    def generate_with_openrouter_safely(self, messages, generation_parameters):
        payload = {
            "model": self.model_name,
            "messages": messages,
        }
        if MAX_TOKENS_KEY in generation_parameters:
            mt = generation_parameters[MAX_TOKENS_KEY]
            # Some providers (e.g. Azure-hosted GPT-5.x) enforce a min max_tokens (>= 16).
            floor = min_max_tokens_for(self.model_name)
            if floor and mt < floor:
                mt = floor
            payload["max_tokens"] = mt
        if TEMPERATURE_KEY in generation_parameters and model_supports_temperature(self.model_name):
            payload["temperature"] = generation_parameters[TEMPERATURE_KEY]

        # Reasoning ("thinking") models default to a high effort on OpenRouter, which
        # is slow and burns the small chat token budget on internal reasoning. Force
        # LOW effort for GPT-5.x / o-series / Gemini Pro (Claude is left untouched -
        # its extended thinking is off by default).
        _reasoning = reasoning_payload_for(self.model_name)
        if _reasoning:
            payload["reasoning"] = _reasoning

        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        output = None
        backoff = SLEEPING_TIME_FOR_API_GENERATION_ERROR
        network_failures = 0
        total_attempts = 0
        MAX_NETWORK_FAILURES = 5
        MAX_TOTAL_ATTEMPTS = 15
        while not output:
            total_attempts += 1
            if total_attempts > MAX_TOTAL_ATTEMPTS:
                raise RuntimeError(
                    f"OpenRouter model '{self.model_name}' failed after {MAX_TOTAL_ATTEMPTS} total attempts"
                )
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                # Defensive: if the provider rejects `temperature` for this model, strip and retry.
                if response.status_code == 400 and "temperature" in payload and \
                        is_unsupported_param_error(response.status_code, response.text):
                    self.logger.log("openrouter_unsupported_param",
                                    f"Stripping temperature for {self.model_name}")
                    payload.pop("temperature", None)
                    continue  # retry without temperature
                # Defensive: if the provider rejects `reasoning`, strip and retry.
                if response.status_code == 400 and "reasoning" in payload and \
                        "reasoning" in response.text.lower():
                    self.logger.log("openrouter_unsupported_param",
                                    f"Stripping reasoning for {self.model_name}")
                    payload.pop("reasoning", None)
                    continue  # retry without reasoning
                # Defensive: if the provider rejects max_tokens as too low, bump it & retry.
                if response.status_code == 400 and "max_tokens" in payload and \
                        is_max_tokens_too_low_error(response.status_code, response.text):
                    new_max = max(int(payload["max_tokens"]) * 2, 16)
                    self.logger.log("openrouter_max_tokens_bump",
                                    f"{self.model_name}: bumping max_tokens {payload['max_tokens']} -> {new_max}")
                    payload["max_tokens"] = new_max
                    continue
                response.raise_for_status()
                rj = response.json()
                output = rj["choices"][0]["message"]["content"]
                usage = rj.get("usage", {})
                if usage:
                    it, ot = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
                    self._last_usage = {"input_tokens": it, "output_tokens": ot,
                                        "estimated_cost_usd": self._estimate_cost(self.model_name, it, ot)}
                backoff = SLEEPING_TIME_FOR_API_GENERATION_ERROR  # reset on success
                network_failures = 0
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    # Rate limited - back off progressively (free tier is strict)
                    print(f"OpenRouter 429 rate limit — waiting {backoff}s")
                    self.logger.log("openrouter_rate_limit", f"429 — sleeping {backoff}s")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 120)  # exponential up to 2 min
                else:
                    print(f"OpenRouter error: {e}")
                    self.logger.log("error generating with OpenRouter", str(e))
                    time.sleep(SLEEPING_TIME_FOR_API_GENERATION_ERROR)
            except Exception as e:
                network_failures += 1
                print(f"OpenRouter error: {e}")
                self.logger.log("error generating with OpenRouter", str(e))
                if network_failures >= MAX_NETWORK_FAILURES:
                    raise RuntimeError(
                        f"OpenRouter unreachable after {MAX_NETWORK_FAILURES} consecutive failures: {e}"
                    )
                time.sleep(SLEEPING_TIME_FOR_API_GENERATION_ERROR)
        return output

    def generate_with_gemini(self, input_text, system_info, generation_parameters):
        if system_info:
            full_prompt = system_info + "\n\n" + input_text
        else:
            full_prompt = input_text

        # Build GenerationConfig from the supplied token and temperature settings.
        max_tokens = generation_parameters.get(MAX_NEW_TOKENS_KEY)
        temperature = generation_parameters.get(TEMPERATURE_KEY)
        config_kwargs = {}
        # Gemini thinking-model token budget strategy:
        #   Gemini 3 (3.x):  use thinking_level ("low"/"high"), not thinking_budget.
        #                    Pro can't fully disable thinking, so use the lowest level to
        #                    cut latency. Applies to pro AND flash/flash-lite - Gemini 3
        #                    ignores thinking_budget entirely.
        #   Gemini 2.5 Flash: thinking can be disabled (budget=0). Keep max_output_tokens modest.
        #   Gemini 2.5 Pro:   thinking is REQUIRED - minimum budget is 128. Needs more total
        #                     tokens to leave room for actual text after thinking.
        model_lower = self.model_name.lower()
        is_gemini_3 = "gemini-3" in model_lower
        is_pro = "pro" in model_lower and "flash" not in model_lower
        if max_tokens is not None:
            base = int(max_tokens)
            if is_pro:
                floor = 300
            elif is_gemini_3:
                floor = 200  # "low" thinking still consumes part of the output budget
            else:
                floor = 100
            config_kwargs["max_output_tokens"] = max(base * 4, floor)
        if temperature is not None:
            config_kwargs["temperature"] = float(temperature)
        if is_gemini_3:
            # Gemini 3: lowest thinking level (no thinking_budget support).
            config_kwargs["thinking_config"] = genai_types.ThinkingConfig(thinking_level="low")
        elif is_pro:
            # Gemini 2.5 Pro requires thinking; use the minimum allowed budget (128) to keep it brief.
            config_kwargs["thinking_config"] = genai_types.ThinkingConfig(thinking_budget=128)
        else:
            # Gemini 2.5 Flash: disable thinking entirely - no value for 2-6 word chat messages.
            config_kwargs["thinking_config"] = genai_types.ThinkingConfig(thinking_budget=0)
        # Disable safety filters - this is a research project where the AI plays a social
        # deception game; without BLOCK_NONE Gemini filters the generator's output as
        # "impersonation / deceptive content" and returns response.text = None.
        config_kwargs["safety_settings"] = [
            genai_types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",        threshold="BLOCK_NONE"),
            genai_types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",        threshold="BLOCK_NONE"),
            genai_types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT",  threshold="BLOCK_NONE"),
            genai_types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT",  threshold="BLOCK_NONE"),
        ]
        gen_config = genai_types.GenerateContentConfig(**config_kwargs)

        # The Google genai SDK retries some errors internally via tenacity, but
        # network-level failures (DNS, server disconnect) are not reliably covered.
        # Wrap the call with our own retry for those transient connection errors.
        _NETWORK_ERROR_KEYWORDS = ("getaddrinfo", "RemoteProtocol", "ConnectError", "ConnectionError", "Server disconnected", "UNAVAILABLE", "503")
        _RATE_LIMIT_KEYWORDS = ("429", "RESOURCE_EXHAUSTED", "rate limit", "quota exceeded", "Too Many Requests")
        response = None
        for _attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=full_prompt,
                    config=gen_config,
                )
                break
            except Exception as _e:
                err_str = str(_e)
                is_rate_limit = any(k in err_str for k in _RATE_LIMIT_KEYWORDS)
                is_network = any(k in err_str for k in _NETWORK_ERROR_KEYWORDS)
                if _attempt < 2 and (is_rate_limit or is_network):
                    # Short, jittered backoff. This call path is shared by the
                    # scheduler, generator and reviewer, so a long sleep here would
                    # freeze the whole turn - the agent would vanish for a minute,
                    # which is both more suspicious and more disruptive than skipping
                    # one turn and retrying on the next game-loop tick. We therefore
                    # cap the backoff at a few seconds and let the caller recover.
                    _wait = min(2 ** _attempt * 2, 8) + random.uniform(0.0, 0.5)
                    if is_rate_limit:
                        self.logger.log("AGENT_ERROR", f"RATE_LIMIT: {_e}")
                    kind = "rate limited" if is_rate_limit else "transient error"
                    print(f"[WARNING] Gemini {kind} (attempt {_attempt + 1}/3), retrying in {_wait:.1f}s: {_e}")
                    time.sleep(_wait)
                else:
                    raise

        # response.text can be None when the response is blocked or when all parts are
        # "thinking" tokens (Gemini 2.5 reasoning models).  Fall back to iterating parts.
        text = response.text if hasattr(response, "text") else None
        if text is None:
            try:
                parts = response.candidates[0].content.parts
                text = "".join(p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False))
            except (AttributeError, IndexError, TypeError):
                pass
        if not text:
            self.logger.log("gemini_empty_response", str(response))
            print(f"[WARNING] Gemini returned empty/blocked response: {response}")
            text = ""
        try:
            um = response.usage_metadata
            it = getattr(um, 'prompt_token_count', 0) or 0
            ot = getattr(um, 'candidates_token_count', 0) or 0
            self._last_usage = {"input_tokens": it, "output_tokens": ot,
                                 "estimated_cost_usd": self._estimate_cost(self.model_name, it, ot)}
        except Exception:
            pass
        return text
