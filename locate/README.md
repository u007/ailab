# Locate

Serve the [nvidia/LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) model using Transformers with an OpenAI-compatible API.

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

## Setup

```bash
# Install dependencies
make install

# Pre-download the model (~6GB)
make download

# Start the API server (port 8282)
make serve
```

## API

The server exposes an OpenAI-compatible API at `http://localhost:8282`.

### Endpoints

- `GET /v1/models` — List available models
- `GET /health` — Health check
- `POST /v1/chat/completions` — Chat completions (supports images)

### Example

```bash
curl -X POST "http://localhost:8282/v1/chat/completions" \
	-H "Content-Type: application/json" \
	--data '{
		"model": "nvidia/LocateAnything-3B",
		"messages": [
			{
				"role": "user",
				"content": [
					{
						"type": "text",
						"text": "Describe this image in one sentence."
					},
					{
						"type": "image_url",
						"image_url": {
							"url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
						}
					}
				]
			}
		]
	}'

```

## Notes

- The model uses Hugging Face Transformers with `trust_remote_code=True`
- Video loading requires `decord` (not available on macOS ARM64); image inference works fine without it
- First request downloads the model (~6GB)
