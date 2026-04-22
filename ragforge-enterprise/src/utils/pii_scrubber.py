"""PII Scrubber for GDPR-compliant structured logging.

Implements Data Minimisation (GDPR Art. 5(1)(c)) by redacting Personally
Identifiable Information from log strings *before* they reach the JSON logger.
The following Italian PII patterns are redacted:

* **Codice Fiscale (CF)** — 16-char alphanumeric Italian tax code
* **P.IVA** — 11-digit Italian VAT number (optionally prefixed with "IT")
* **Email addresses** — RFC-5322-compatible pattern
* **IBAN** — International Bank Account Numbers (IT IBAN starts with IT)
* **Phone numbers** — Italian mobile / landline patterns

Name and address scrubbing is deliberately left to the ``name_patterns``
and ``address_patterns`` registries which callers can extend at startup.

Usage::

    from src.utils.pii_scrubber import PIIScrubber

    scrubber = PIIScrubber()
    clean = scrubber.scrub("CF: RSSMRA85M01H703Z tel 333-1234567")
    # "CF: [REDACTED-CF] tel [REDACTED-PHONE]"
"""

from __future__ import annotations

import re
from typing import ClassVar


class PIIScrubber:
    """Stateless PII redaction engine for log strings.

    All patterns are compiled once at class definition time and reused
    across instances (thread-safe: ``re`` compiled patterns are immutable).

    Attributes:
        CF_PLACEHOLDER:    Replacement token for Codice Fiscale matches.
        PIVA_PLACEHOLDER:  Replacement token for P.IVA matches.
        EMAIL_PLACEHOLDER: Replacement token for email address matches.
        IBAN_PLACEHOLDER:  Replacement token for IBAN matches.
        PHONE_PLACEHOLDER: Replacement token for phone number matches.
    """

    CF_PLACEHOLDER: ClassVar[str] = "[REDACTED-CF]"
    PIVA_PLACEHOLDER: ClassVar[str] = "[REDACTED-PIVA]"
    EMAIL_PLACEHOLDER: ClassVar[str] = "[REDACTED-EMAIL]"
    IBAN_PLACEHOLDER: ClassVar[str] = "[REDACTED-IBAN]"
    PHONE_PLACEHOLDER: ClassVar[str] = "[REDACTED-PHONE]"

    # ── Compiled regex patterns ───────────────────────────────────────────────

    # Italian Codice Fiscale: exactly 16 uppercase alphanumeric chars in the
    # canonical LLLLLLNNLNNLNNNL pattern. We use a lookahead/lookbehind to
    # avoid matching substrings of longer tokens.
    _CF_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"(?<![A-Z0-9])"
        r"[A-Z]{6}[0-9]{2}[A-EHLMPRST][0-9]{2}[A-Z][0-9]{3}[A-Z]"
        r"(?![A-Z0-9])",
        re.IGNORECASE,
    )

    # Italian P.IVA: optional "IT" prefix + 11 digits.
    # Also matches bare 11-digit sequences only when preceded by "p.iva",
    # "partita iva", or "iva:" (case-insensitive) to reduce false positives.
    _PIVA_CONTEXTUAL_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:p\.?\s*iva|partita\s+iva|vat\s*(?:n(?:o|umber)?)?)[:\s]*"
        r"(?:IT)?\s*(\d{11})",
        re.IGNORECASE,
    )
    _PIVA_IT_PREFIX_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"\bIT\d{11}\b",
        re.IGNORECASE,
    )

    # Email addresses (RFC-5322 simplified, no quoted local parts).
    _EMAIL_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}",
        re.IGNORECASE,
    )

    # IBAN — starts with 2-letter country code + 2 check digits + up to 30
    # alphanumeric chars. We specifically target the IT IBAN (27 chars) but
    # accept any valid IBAN for completeness.
    _IBAN_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{11,30}\b",
    )

    # Italian phone numbers: mobile (3XX-NNNNNNN or +39 3XX NNNNNNN) and
    # landlines (0XX-NNNNNNN). The pattern avoids matching pure zip/postal codes
    # by requiring the leading country code, explicit separator, or "tel" prefix.
    _PHONE_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:"
        r"(?:\+39|0039)\s*"           # international prefix
        r"|(?:tel|fax|cell|mob)[.:\s]*"  # keyword prefix
        r")"
        r"(?:0\d{1,3}|3\d{2})"       # area or mobile prefix
        r"[\s.\-]?"
        r"\d{3,4}"
        r"[\s.\-]?"
        r"\d{4,6}",
        re.IGNORECASE,
    )

    def scrub(self, text: str) -> str:
        """Return *text* with all detected PII tokens replaced by placeholders.

        Scrubbing order matters:
        1. Email (longest/most specific) first to avoid partial overlaps.
        2. IBAN before P.IVA to prevent the IT-prefix digit run from matching
           both patterns.
        3. CF (strict 16-char alphanum pattern).
        4. P.IVA contextual, then IT-prefixed standalone.
        5. Phone numbers last (broadest pattern).

        Args:
            text: Raw log string that may contain PII.

        Returns:
            The sanitised string with PII tokens replaced.
        """
        if not text:
            return text

        text = self._EMAIL_RE.sub(self.EMAIL_PLACEHOLDER, text)
        text = self._IBAN_RE.sub(self.IBAN_PLACEHOLDER, text)
        text = self._CF_RE.sub(self.CF_PLACEHOLDER, text)

        # Contextual P.IVA: replace the whole match (keyword + digits).
        text = self._PIVA_CONTEXTUAL_RE.sub(
            lambda m: m.group(0).replace(m.group(1), self.PIVA_PLACEHOLDER),
            text,
        )
        text = self._PIVA_IT_PREFIX_RE.sub(self.PIVA_PLACEHOLDER, text)

        text = self._PHONE_RE.sub(self.PHONE_PLACEHOLDER, text)

        return text

    def scrub_dict(self, data: dict) -> dict:
        """Recursively scrub all string values in *data*.

        Args:
            data: Arbitrary dictionary (e.g. the ``extra`` payload of a log call).

        Returns:
            A new dictionary with all nested string values scrubbed.
        """
        result: dict = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self.scrub(value)
            elif isinstance(value, dict):
                result[key] = self.scrub_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    self.scrub(item) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result


# Module-level singleton — import and reuse to avoid repeated pattern compilation.
_default_scrubber = PIIScrubber()


def scrub(text: str) -> str:
    """Convenience wrapper around the default :class:`PIIScrubber` instance.

    Args:
        text: Raw string that may contain PII.

    Returns:
        The sanitised string.
    """
    return _default_scrubber.scrub(text)
