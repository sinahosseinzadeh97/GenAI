"""LexReview — Legal document analysis sub-system for RAGForge Enterprise.

Sub-packages
------------
extraction
    spaCy NER + regex extraction of Clause and LegalEntities objects.
agent
    LegalRAGAgent: HybridRetriever → Reranker → LLM (CoT) pipeline.
eval
    Retrieval + faithfulness metrics and RAGEvaluator harness.
finetune
    Colab-ready LoRA/QLoRA data preparation and SFTTrainer wrapper.
api
    FastAPI router exposing /query, /extract, /index, /search endpoints.
"""

__all__: list[str] = []
