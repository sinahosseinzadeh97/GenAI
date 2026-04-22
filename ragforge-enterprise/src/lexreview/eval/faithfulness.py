"""LLM-as-judge faithfulness evaluator.

:class:`FaithfulnessJudge` calls an LLM to score how well an agent answer
is supported by the retrieved context passages.  The judge prompt uses a
rubric that returns a numeric score in [0.0, 1.0].

Typical usage::

    judge = FaithfulnessJudge(llm_client)
    score = judge.score(
        answer="The governing law is Delaware.",
        context_chunks=["...", "..."],
    )
"""

from __future__ import annotations

import re

from src.lexreview.agent.llm_client import LLMClient
from src.utils.logger import get_logger

log = get_logger(__name__)

_FAITHFULNESS_PROMPT = """\
You are an impartial legal document evaluator. Your task is to assess whether
a given ANSWER is faithfully supported by the provided CONTEXT.

CONTEXT:
{context}

ANSWER:
{answer}

Evaluation rubric:
- 1.0 = Every claim in the answer is directly supported by the context.
- 0.75 = Most claims are supported; minor extrapolations.
- 0.5 = Some claims are supported but others lack grounding.
- 0.25 = Only a few claims have contextual support.
- 0.0 = The answer contradicts or ignores the context entirely.

Respond with ONLY a JSON object in this exact format:
{{"score": <float between 0.0 and 1.0>, "reason": "<one sentence explanation>"}}
"""

_SCORE_RE = re.compile(r'"score"\s*:\s*([0-9]*\.?[0-9]+)')


class FaithfulnessJudge:
    """LLM-as-judge for rating answer faithfulness to retrieved context.

    Args:
        llm: :class:`~src.lexreview.agent.llm_client.LLMClient` instance.
            If ``None``, a default client is created.
        min_faithful_threshold: Score at or above which ``is_faithful`` is True.

    Example::

        judge = FaithfulnessJudge(llm_client)
        result = judge.score("Parties may terminate with 30 days notice.", chunks)
        print(result["score"], result["is_faithful"])
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        min_faithful_threshold: float = 0.5,
        max_context_chunks: int = 8,
    ) -> None:
        self._llm = llm if llm is not None else LLMClient()
        self._threshold = min_faithful_threshold
        self._max_context_chunks = max_context_chunks

    def score(
        self,
        answer: str,
        context_chunks: list[str],
    ) -> dict[str, object]:
        """Judge how faithfully *answer* is grounded in *context_chunks*.

        Args:
            answer:         Agent's synthesised answer string.
            context_chunks: List of raw chunk texts used to generate the answer.

        Returns:
            Dict with keys:
            - ``"score"`` (float in [0.0, 1.0])
            - ``"reason"`` (str explanation from the judge LLM)
            - ``"is_faithful"`` (bool)
        """
        if len(context_chunks) > self._max_context_chunks:
            log.warning(
                "FaithfulnessJudge: context truncated",
                extra={
                    "provided": len(context_chunks),
                    "used": self._max_context_chunks,
                    "discarded": len(context_chunks) - self._max_context_chunks,
                },
            )
        context = "\n---\n".join(c.strip() for c in context_chunks[:self._max_context_chunks])
        prompt = _FAITHFULNESS_PROMPT.format(context=context, answer=answer)

        try:
            raw = self._llm.complete_text(prompt)
            score = self._parse_score(raw)
            reason = self._parse_reason(raw)
        except Exception as exc:
            log.warning(
                "FaithfulnessJudge scoring failed, defaulting to 0.0",
                extra={"error": str(exc)},
            )
            score = 0.0
            reason = f"Judge call failed: {exc}"

        result: dict[str, object] = {
            "score": round(score, 4),
            "reason": reason,
            "is_faithful": score >= self._threshold,
        }
        log.debug("FaithfulnessJudge result", extra=result)
        return result

    def _parse_score(self, raw: str) -> float:
        """Extract the numeric score from the judge's JSON response."""
        match = _SCORE_RE.search(raw)
        if match:
            return max(0.0, min(1.0, float(match.group(1))))
        return 0.0

    def _parse_reason(self, raw: str) -> str:
        """Extract the reason string from the judge's JSON response."""
        reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', raw)
        return reason_match.group(1) if reason_match else raw.strip()[:200]
