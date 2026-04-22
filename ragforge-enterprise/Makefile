# ─────────────────────────────────────────────────────────────────────────────
# RAGForge Enterprise – Makefile
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: help check-secrets

# Patterns that indicate a real OpenAI key is present (ERE syntax)
SECRET_PATTERNS := sk-proj-[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9]{20,}

# ── help ──────────────────────────────────────────────────────────────────────
help: ## Show available Make targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ── check-secrets ─────────────────────────────────────────────────────────────
check-secrets: ## Scan all tracked files + .env for exposed OpenAI API keys
	@echo "🔍  Scanning git-tracked files for secret patterns…"
	@MATCHES=$$(git ls-files | xargs grep -El "$(SECRET_PATTERNS)" 2>/dev/null); \
	 ENV_MATCH=$$(grep -El "$(SECRET_PATTERNS)" .env 2>/dev/null); \
	 ALL_MATCHES="$$MATCHES $$ENV_MATCH"; \
	 ALL_MATCHES=$$(echo "$$ALL_MATCHES" | tr ' ' '\n' | sort -u | grep -v '^$$'); \
	 if [ -n "$$ALL_MATCHES" ]; then \
	   echo ""; \
	   echo "❌  SECRET(S) FOUND in the following file(s):"; \
	   echo "$$ALL_MATCHES" | sed 's/^/   /'; \
	   echo ""; \
	   echo "   Replace real keys with placeholders and inject via Doppler,"; \
	   echo "   1Password CLI, or direnv before committing."; \
	   echo ""; \
	   exit 1; \
	 else \
	   echo "✅  No exposed secrets found."; \
	 fi
