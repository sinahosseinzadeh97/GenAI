"""
WARNING: The `relevant_chunk_ids` in these samples are synthetic placeholders
(e.g. "chunk-001"). They will never match real chunk UUIDs in a Qdrant collection.
As a result, precision@k, recall@k, and MRR will always be 0.0 when these samples
are used against a live index.

To use real evaluation, generate ground-truth samples against an actual indexed
collection and replace the `relevant_chunk_ids` values with real Qdrant point IDs.
See: src/lexreview/eval/README.md (to be created) for instructions.

These samples cover the 9 core legal clause categories and can be used
for regression testing, pipeline smoke-testing, and metric baselining
without a live document store.
"""

from __future__ import annotations

from src.lexreview.eval.models import EvalSample

EVAL_SAMPLES: list[EvalSample] = [
    EvalSample(
        sample_id="eval-001",
        question="What are the payment terms specified in this agreement?",
        ground_truth_answer=(
            "Payment is due within 30 days of invoice receipt.  Late payments "
            "accrue interest at 1.5% per month."
        ),
        relevant_chunk_ids=["chunk-payment-01", "chunk-payment-02"],  # PLACEHOLDER: replace with real Qdrant point IDs from your indexed collection
        category="payment",
    ),
    EvalSample(
        sample_id="eval-002",
        question="Under what circumstances can either party terminate this contract?",
        ground_truth_answer=(
            "Either party may terminate the agreement upon 30 days written notice, "
            "or immediately for cause upon material breach."
        ),
        relevant_chunk_ids=["chunk-termination-01", "chunk-termination-02"],  # PLACEHOLDER: replace with real Qdrant point IDs from your indexed collection
        category="termination",
    ),
    EvalSample(
        sample_id="eval-003",
        question="What is the indemnification obligation of the vendor?",
        ground_truth_answer=(
            "The vendor shall indemnify, defend, and hold harmless the client from "
            "any third-party claims arising from vendor's breach of this Agreement."
        ),
        relevant_chunk_ids=["chunk-indemnity-01"],  # PLACEHOLDER: replace with real Qdrant point IDs from your indexed collection
        category="indemnification",
    ),
    EvalSample(
        sample_id="eval-004",
        question="What is the maximum liability cap under this agreement?",
        ground_truth_answer=(
            "In no event shall either party's aggregate liability exceed the total "
            "fees paid in the twelve months preceding the claim."
        ),
        relevant_chunk_ids=["chunk-liability-01", "chunk-liability-02"],  # PLACEHOLDER: replace with real Qdrant point IDs from your indexed collection
        category="limitation_of_liability",
    ),
    EvalSample(
        sample_id="eval-005",
        question="What governing law and jurisdiction applies to disputes?",
        ground_truth_answer=(
            "This Agreement is governed by the laws of the State of Delaware. "
            "Any disputes shall be resolved exclusively in the courts of New Castle County."
        ),
        relevant_chunk_ids=["chunk-governing-law-01"],  # PLACEHOLDER: replace with real Qdrant point IDs from your indexed collection
        category="governing_law",
    ),
    EvalSample(
        sample_id="eval-006",
        question="What confidentiality obligations apply to the receiving party?",
        ground_truth_answer=(
            "The receiving party shall not disclose confidential information to "
            "any third party and shall use it only for the purposes of this Agreement."
        ),
        relevant_chunk_ids=["chunk-confidentiality-01", "chunk-confidentiality-02"],  # PLACEHOLDER: replace with real Qdrant point IDs from your indexed collection
        category="confidentiality",
    ),
    EvalSample(
        sample_id="eval-007",
        question="What events constitute a force majeure under this contract?",
        ground_truth_answer=(
            "Force majeure events include acts of God, natural disasters, government "
            "actions, pandemics, and other events beyond the reasonable control of a party."
        ),
        relevant_chunk_ids=["chunk-force-majeure-01"],  # PLACEHOLDER: replace with real Qdrant point IDs from your indexed collection
        category="force_majeure",
    ),
    EvalSample(
        sample_id="eval-008",
        question="How must disputes be resolved — through arbitration or litigation?",
        ground_truth_answer=(
            "All disputes shall be submitted to binding arbitration under the rules "
            "of the American Arbitration Association in New York, New York."
        ),
        relevant_chunk_ids=["chunk-arbitration-01"],  # PLACEHOLDER: replace with real Qdrant point IDs from your indexed collection
        category="dispute_resolution",
    ),
    EvalSample(
        sample_id="eval-009",
        question="What warranties does the software vendor provide?",
        ground_truth_answer=(
            "The vendor warrants that the software will perform substantially in "
            "accordance with documentation for 90 days following delivery."
        ),
        relevant_chunk_ids=["chunk-warranty-01", "chunk-warranty-02"],  # PLACEHOLDER: replace with real Qdrant point IDs from your indexed collection
        category="warranty",
    ),
    EvalSample(
        sample_id="eval-010",
        question="What are the notice requirements under this agreement?",
        ground_truth_answer=(
            "Notices must be in writing and delivered by certified mail or overnight "
            "courier to the addresses specified in Exhibit A."
        ),
        relevant_chunk_ids=["chunk-notice-01"],  # PLACEHOLDER: replace with real Qdrant point IDs from your indexed collection
        category="general",
    ),
]
