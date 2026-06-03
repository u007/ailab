.PHONY: install download serve clean help

help:
	@echo "Usage:"
	@echo "  make install   - Install dependencies using uv"
	@echo "  make download  - Pre-download the LocateAnything-3B model"
	@echo "  make serve     - Start LocateAnything API server (port 8282)"
	@echo "  make clean     - Remove virtual environment"

install:
	uv sync

download:
	uv run huggingface-cli download nvidia/LocateAnything-3B

serve:
	LOCATE_PORT=8282 uv run locateanything-api

clean:
	rm -rf .venv
