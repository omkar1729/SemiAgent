"""LLM-as-judge rubric for scoring diagnosis reports (1-5 per dimension).

The five rubric dimensions are FROZEN once any scoring run has started
(HARD RULE 8) — do not edit them after results exist.

All five dimensions are scored in ONE batched call (returning a JSON object of the
five scores) rather than five separate calls. This is purely an efficiency change —
~5x fewer requests and far fewer tokens, which keeps us under the OpenAI TPM cap so
calls don't silently back off — and the dimension definitions are unchanged.
"""
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openai

from config import settings

log = logging.getLogger("rubric")

RUBRIC_DIMENSIONS = [
    {
        "key": "causal_correctness",
        "name": "Causal Correctness",
        "definition": "Does the hypothesis identify a plausible process-level cause for the defect pattern?",
        "excellent": "Hypotheses name specific, plausible process-level causes that match the defect morphology.",
        "acceptable": "Hypotheses are generally plausible but partly generic or loosely matched to the pattern.",
        "poor": "Hypotheses are implausible, absent, or unrelated to the defect pattern.",
    },
    {
        "key": "evidence_grounding",
        "name": "Evidence Grounding",
        "definition": "Is each hypothesis explicitly supported by cited retrieved document text?",
        "excellent": "Every hypothesis cites and is consistent with specific retrieved documents.",
        "acceptable": "Some hypotheses cite documents; grounding is partial or uneven.",
        "poor": "No meaningful citations or citations contradict the retrieved text.",
    },
    {
        "key": "technical_accuracy",
        "name": "Technical Accuracy",
        "definition": "Are semiconductor process terms and mechanisms used correctly?",
        "excellent": "Process terminology and mechanisms are used correctly throughout.",
        "acceptable": "Mostly correct with minor terminology or mechanism errors.",
        "poor": "Frequent or serious technical errors.",
    },
    {
        "key": "completeness",
        "name": "Completeness",
        "definition": "Are multiple distinct causal pathways considered?",
        "excellent": "Several distinct, non-overlapping causal pathways are considered.",
        "acceptable": "Two pathways considered, with some overlap.",
        "poor": "Only one pathway or none.",
    },
    {
        "key": "actionability",
        "name": "Actionability",
        "definition": "Are concrete measurable process parameters identified for investigation?",
        "excellent": "Specific, measurable parameters are listed for each hypothesis.",
        "acceptable": "Some concrete parameters, but vague or incomplete.",
        "poor": "No actionable or measurable parameters.",
    },
]

# Fields of the diagnosis dict that the judge needs. The raw chain-of-thought
# (`reasoning_chain`) is excluded — it is large and redundant with the structured
# fields, and dropping it roughly halves the per-call token count.
_SCORED_FIELDS = ["defect_summary", "process_step_analysis",
                  "root_cause_hypotheses", "investigation_parameters"]


class DiagnosticRubric:
    def __init__(self):
        # generous retries so transient 429 rate-limit bursts back off, not crash;
        # explicit timeout so a stalled connection fails fast and retries (not a 600s hang)
        self.client = openai.OpenAI(api_key=settings.openai_api_key, max_retries=8, timeout=60.0)

    def _prompt(self, diagnosis_dict, docs_block) -> str:
        dims = "\n".join(
            f"- {d['key']} ({d['name']}): {d['definition']}\n"
            f"    5=Excellent: {d['excellent']}\n"
            f"    3=Acceptable: {d['acceptable']}\n"
            f"    1=Poor: {d['poor']}"
            for d in RUBRIC_DIMENSIONS)
        keys = ", ".join(f'"{d["key"]}"' for d in RUBRIC_DIMENSIONS)
        report = {k: diagnosis_dict.get(k) for k in _SCORED_FIELDS}
        return f"""You are evaluating a semiconductor defect root cause analysis report on five dimensions, each scored 1-5 (5 best, 1 worst).

Dimensions:
{dims}

Diagnosis report:
{json.dumps(report, indent=2)}

Retrieved documents used:
{docs_block}

Respond with ONLY a JSON object mapping each of these keys to its integer score (1-5): {keys}."""

    def auto_score(self, diagnosis_dict: dict, retrieved_docs: list) -> dict:
        """Score all five dimensions in one call; `overall` is the mean of the five
        (1-5) rescaled to 0-10 (mean x 2)."""
        docs_block = "\n".join((d.get("text", "")[:300]) for d in retrieved_docs) or "[none]"
        prompt = self._prompt(diagnosis_dict, docs_block)

        scores = None
        for _ in range(3):  # parse, retry twice
            resp = self.client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            try:
                data = json.loads(resp.choices[0].message.content or "{}")
                parsed = {d["key"]: int(data[d["key"]]) for d in RUBRIC_DIMENSIONS}
                scores = {k: min(5, max(1, v)) for k, v in parsed.items()}
                break
            except Exception:
                continue
        if scores is None:
            log.warning("Could not parse rubric JSON; defaulting all dimensions to 1.")
            scores = {d["key"]: 1 for d in RUBRIC_DIMENSIONS}

        mean_5 = sum(scores.values()) / len(RUBRIC_DIMENSIONS)
        scores["overall"] = round(mean_5 * 2.0, 2)  # report out of 10
        return scores
