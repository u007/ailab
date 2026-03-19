PYTHON_VERSION := 3.12.9
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
MODEL_DIR := models/GLM-OCR-bf16
HOST := 0.0.0.0
PORT := 8081
WEBUI_PORT := 8080
MAX_TOKENS := 4096

.PHONY: serve download install check setup shell webui webui-install

setup:
	pyenv install -s $(PYTHON_VERSION)
	pyenv local $(PYTHON_VERSION)
	$$(pyenv which python) -m venv $(VENV)
	$(PIP) install -U pip

install: setup
	$(PIP) install -U mlx-vlm open-webui

serve: check
	$(PYTHON) -m mlx_vlm.server --host $(HOST) --port $(PORT)

generate: check
	$(PYTHON) -m mlx_vlm.generate \
		--model $(MODEL_DIR) \
		--max-tokens $(MAX_TOKENS) \
		--temperature 0.0 \
		--prompt "OCR the text in this image." \
		--image $(IMAGE)

webui:
	OPENAI_API_BASE_URL=http://localhost:$(PORT)/v1 \
	DATA_DIR=./data \
	$(VENV)/bin/open-webui serve --port $(WEBUI_PORT)

shell:
	@. $(VENV)/bin/activate && exec $(SHELL)

download:
	huggingface-cli download mlx-community/GLM-OCR-bf16 --local-dir $(MODEL_DIR)

check:
	@test -d $(VENV) || (echo "Venv not found. Run 'make install' first." && exit 1)
	@test -d $(MODEL_DIR) || (echo "Model not found. Run 'make download' first." && exit 1)
	@$(PYTHON) -c "import mlx_vlm" 2>/dev/null || (echo "mlx-vlm not installed. Run 'make install' first." && exit 1)
