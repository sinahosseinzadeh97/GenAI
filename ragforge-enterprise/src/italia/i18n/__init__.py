"""RAGForge Italia — i18n sub-package.

Provides Italian-language error message catalogues and localisation utilities.

Usage::

    from src.italia.i18n import get_error_message

    msg = get_error_message("NOT_FOUND", lang="it")
    # → "Risorsa non trovata."
"""

from src.italia.i18n.errors_it import ERROR_MESSAGES_IT, get_error_message_it

__all__ = [
    "ERROR_MESSAGES_IT",
    "get_error_message_it",
]
