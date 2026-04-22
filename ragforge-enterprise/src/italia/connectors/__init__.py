"""Connector registry for RAGForge Italia.

All connector classes are registered here so the CLI and pipeline can
discover them by name without importing each module individually.

Usage::

    from src.italia.connectors import CONNECTORS

    connector = CONNECTORS["normattiva"]()
    docs = connector.fetch(codice="civile")

Phase 6 — Integration connectors (FileNet, LexisNexis, Notartel, SIECIC/SICID)
are registered separately under ``INTEGRATION_CONNECTORS`` because they require
credentials and are optional for deployments without those integrations.

    from src.italia.connectors import INTEGRATION_CONNECTORS

    connector = INTEGRATION_CONNECTORS["notartel"](token="...", stub=True)
    docs = connector.fetch()

Webhook endpoints for push-based integrations::

    from src.italia.connectors import webhook_router
    app.include_router(webhook_router, prefix="/italia")
"""

from src.italia.connectors.agcm import AGCMConnector
from src.italia.connectors.bancaditalia import BancaItaliaConnector
from src.italia.connectors.cassazione import CassazioneConnector
from src.italia.connectors.corte_costituzionale import CorteCostituzionaleConnector
from src.italia.connectors.dejure import DeJureConnector
from src.italia.connectors.eurlex import EurLexConnector
from src.italia.connectors.filenet_documentum import FilenetDocumentumConnector
from src.italia.connectors.gazzetta import GazzettaUfficialeConnector
from src.italia.connectors.lexisnexis_it import LexisNexisItaliaConnector
from src.italia.connectors.normattiva import NormativaConnector
from src.italia.connectors.notartel import NotartelConnector
from src.italia.connectors.siecic_sicid import SiecicSicidConnector
from src.italia.connectors.tar import TARConnector
from src.italia.connectors.webhook_router import webhook_router

# ── Public-source registry (no credentials required) ──────────────────────────
CONNECTORS: dict[str, type] = {
    "normattiva": NormativaConnector,
    "gazzetta": GazzettaUfficialeConnector,
    "cassazione": CassazioneConnector,
    "eurlex": EurLexConnector,
    "tar": TARConnector,
    "corte_costituzionale": CorteCostituzionaleConnector,
    "agcm": AGCMConnector,
    "bancaditalia": BancaItaliaConnector,
    "dejure": DeJureConnector,
}

# ── Integration connectors (Phase 6 — credentials required) ───────────────────
INTEGRATION_CONNECTORS: dict[str, type] = {
    "filenet": FilenetDocumentumConnector,
    "lexisnexis": LexisNexisItaliaConnector,
    "notartel": NotartelConnector,
    "siecic": SiecicSicidConnector,
    "sicid": SiecicSicidConnector,
}

__all__ = [
    # Registries
    "CONNECTORS",
    "INTEGRATION_CONNECTORS",
    # Webhook router
    "webhook_router",
    # Public-source connectors
    "NormativaConnector",
    "GazzettaUfficialeConnector",
    "CassazioneConnector",
    "EurLexConnector",
    "TARConnector",
    "CorteCostituzionaleConnector",
    "AGCMConnector",
    "BancaItaliaConnector",
    "DeJureConnector",
    # Phase 6 — integration connectors
    "FilenetDocumentumConnector",
    "LexisNexisItaliaConnector",
    "NotartelConnector",
    "SiecicSicidConnector",
]
