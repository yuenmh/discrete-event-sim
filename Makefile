.PHONY: phony

check: phony
	uv tool run ruff format --check
	uv tool run ruff check
	uv tool run basedpyright

fmt: phony
	uv tool run ruff format
