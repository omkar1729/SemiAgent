"""Diagnosis Agent — synthesizes detection, retrieved knowledge, and an optional
engineer note into a structured root cause analysis via GPT-4o-mini with
chain-of-thought prompting."""
import json
import re
import time
from typing import Optional

import openai

from config import settings

SYSTEM_PROMPT = (
    "You are an expert semiconductor process engineer specializing in wafer defect "
    "root cause analysis. Given a detected defect pattern, retrieved technical "
    "knowledge, and optional engineer observations, generate a structured root cause "
    "analysis. Ground your analysis in the retrieved documents. When engineer "
    "observations are provided, incorporate them into your reasoning."
)

_FALLBACK = {
    "defect_summary": "JSON parsing failed",
    "process_step_analysis": "",
    "root_cause_hypotheses": [],
    "investigation_parameters": [],
}


class DiagnosisAgent:
    """Diagnosis via an OpenAI-compatible chat API.

    provider="openai"   -> OpenAI gpt-4o-mini (settings.llm_model).
    provider="semikong"       -> an OpenAI-compatible server (LMStudio/vLLM/Ollama).
    provider="semikong_local" -> SemiKong loaded in-process via transformers +
                                 bitsandbytes 4-bit quantization.
    The SemiKong paths drive the ablation's condition E (domain LLM vs GPT-4o-mini).
    """

    def __init__(self, provider: Optional[str] = None):
        self.provider = (provider or settings.diagnosis_provider).lower()
        self.is_local = self.provider == "semikong_local"
        if self.is_local:
            self._init_local_model()
            self.model = settings.semikong_hf_model
            self.supports_json_mode = False
        elif self.provider == "semikong":
            self.client = openai.OpenAI(
                base_url=settings.semikong_base_url,
                api_key=settings.semikong_api_key or "not-needed",
                max_retries=2, timeout=60.0,
            )
            self.model = settings.semikong_model
            self.supports_json_mode = False  # local servers often lack json_object
        else:
            # generous retries so transient 429 rate-limit bursts back off, not crash;
            # explicit timeout so a stalled connection fails fast and retries (not a 600s hang)
            self.client = openai.OpenAI(api_key=settings.openai_api_key, max_retries=8, timeout=60.0)
            self.model = settings.llm_model
            self.supports_json_mode = True

    def _init_local_model(self):
        """Load SemiKong locally with 4-bit quantization (per the bitsandbytes recipe)."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self._torch = torch
        token = settings.hf_token or None
        bnb = BitsAndBytesConfig(load_in_4bit=True)
        self._tokenizer = AutoTokenizer.from_pretrained(settings.semikong_hf_model, token=token)
        self._hf_model = AutoModelForCausalLM.from_pretrained(
            settings.semikong_hf_model,
            quantization_config=bnb,
            device_map="auto",
            token=token,
        )
        self._hf_model.eval()

    def _chat(self, system_prompt: str, user_prompt: str) -> str:
        """Run one chat completion through the configured backend; return raw text."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if self.is_local:
            torch = self._torch
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._hf_model.device)
            with torch.no_grad():
                out = self._hf_model.generate(
                    **inputs, max_new_tokens=900, do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id)
            return self._tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        kwargs = {"model": self.model, "messages": messages, "temperature": 0.2}
        if self.supports_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        else:
            kwargs["max_tokens"] = 512  # enough for the compact RCA JSON; faster generation
        return self.client.chat.completions.create(**kwargs).choices[0].message.content or ""

    def diagnose(self, detection_result: dict, retrieval_result: dict,
                 engineer_note: Optional[str] = None) -> dict:
        docs = retrieval_result.get("retrieved_docs", [])
        numbered = "\n".join(f"[{i + 1}] {d['text']}" for i, d in enumerate(docs)) or "[none]"

        if engineer_note:
            engineer_context = "Engineer observations: " + engineer_note
        else:
            engineer_context = "No additional engineer observations provided."

        user_prompt = f"""Defect detected: {detection_result.get('defect_type')} (confidence: {detection_result.get('confidence', 0.0):.1%})
Active defect patterns: {detection_result.get('active_defects')}
Detection mode: {detection_result.get('detection_mode')}

{engineer_context}

Technical context from knowledge base:
{numbered}

Think step by step:
Step 1: Describe what this defect pattern looks like spatially on the wafer and what its distribution implies about process origin. If engineer observations were provided, incorporate them here.
Step 2: From the technical context above, identify which semiconductor process steps are most likely associated with this defect morphology and any conditions mentioned in the engineer note.
Step 3: Generate 2 to 3 ranked root cause hypotheses. For each hypothesis cite the supporting document number using [1], [2], etc.
Step 4: List specific process parameters an engineer should measure or adjust to investigate each hypothesis.

Respond with a JSON object using exactly these keys:
- defect_summary: string
- process_step_analysis: string
- root_cause_hypotheses: list of objects each with keys rank (int), hypothesis (string), supporting_evidence (string with document citations), confidence_level (string: High, Medium, or Low)
- investigation_parameters: list of strings"""

        start = time.perf_counter()
        response_text = self._chat(SYSTEM_PROMPT, user_prompt)
        latency_ms = (time.perf_counter() - start) * 1000.0

        parsed = self._parse_json(response_text)
        return {
            "defect_summary": parsed.get("defect_summary", ""),
            "process_step_analysis": parsed.get("process_step_analysis", ""),
            "root_cause_hypotheses": parsed.get("root_cause_hypotheses", []),
            "investigation_parameters": parsed.get("investigation_parameters", []),
            "reasoning_chain": response_text,
            "llm_model_used": self.model,
            "generation_latency_ms": latency_ms,
        }

    @staticmethod
    def _parse_json(response_text: str) -> dict:
        """Parse the LLM response as JSON, with regex + repair fallbacks. Never raises.

        The repair step (json_repair) handles the slightly-malformed JSON that
        smaller models like SemiKong emit (prose prefix, missing commas, etc.) so a
        good answer isn't discarded over formatting."""
        try:
            return json.loads(response_text)
        except Exception:
            pass
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        try:
            from json_repair import repair_json

            repaired = repair_json(response_text, return_objects=True)
            if isinstance(repaired, dict) and repaired:
                return repaired
        except Exception:
            pass
        fallback = dict(_FALLBACK)
        fallback["reasoning_chain"] = response_text
        return fallback
