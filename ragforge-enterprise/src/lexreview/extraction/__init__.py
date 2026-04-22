"""extraction sub-package — spaCy NER, regex, and clause detection."""

from src.lexreview.extraction.clause_detector import ClauseDetector
from src.lexreview.extraction.italian_ner import ItalianLegalEntities, ItalianLegalNER
from src.lexreview.extraction.models import Clause, LegalEntities
from src.lexreview.extraction.ner import LegalNER
from src.lexreview.extraction.regex_extractor import RegexExtractor

__all__ = [
    "Clause",
    "LegalEntities",
    "LegalNER",
    "RegexExtractor",
    "ClauseDetector",
    "ItalianLegalNER",
    "ItalianLegalEntities",
]
