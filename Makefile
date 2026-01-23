.PHONY: setup upgrade install-uv install-tools install-hooks sync

# Install uv if not already installed
install-uv:
	@echo ">>> Installing uv..."
	@if command -v uv > /dev/null 2>&1; then \
		echo "uv is already installed: $$(uv --version)"; \
	else \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		echo "uv installed successfully"; \
	fi

# Install development tools
install-tools:
	@echo ">>> Installing development tools..."
	uv tool install ruff@latest
	uv tool install prek@latest
	uv tool install ty@latest
	uv tool install pyproject-fmt@latest
	uv tool install deptry@latest

# Install pre-commit hooks via prek
install-hooks:
	@echo ">>> Installing pre-commit hooks..."
	uv run prek install

# Sync project dependencies
sync:
	@echo ">>> Syncing project dependencies..."
	uv sync

# Full setup for new environments
setup: install-uv install-tools install-hooks sync
	@echo ""
	@echo "=== Setup complete! ==="

# Upgrade all tools and dependencies
upgrade:
	@echo "=== Upgrading modern-python-project-template ==="
	@echo ""
	@echo ">>> Updating uv..."
	uv self update
	@echo ""
	$(MAKE) install-tools install-hooks sync
	@echo ""
	@echo "=== Upgrade complete! ==="
