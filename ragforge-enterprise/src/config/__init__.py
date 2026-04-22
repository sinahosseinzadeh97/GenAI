"""Configuration sub-package for RAGForge Enterprise.

The canonical way to obtain the validated settings singleton is::

    from src.config import get_settings, Settings

    settings = get_settings()
"""

from src.config.settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
