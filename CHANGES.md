# Changelog

## 2026-06-03

- feat(locate): add LocateAnything API service — FastAPI backend with video frame extraction, object search, and OpenAI-compatible chat completions API
- chore(api): change default port to 8181
- feat(api): add Parakeet transcription FastAPI server
- feat(transcription): add NVIDIA Nemotron and Parakeet ASR support
- fix(validation): handle pixel values and align logits with labels
- feat: add mlx native training support for Apple Silicon

## 2026-05

- feat(download): add resumable download functionality for models and datasets
- feat(vision): add vision model support and dataset download utilities
- feat: add training data and update inference configuration
- feat: add LLM training and inference pipeline with Qwen3-8B model
- feat(claims): implement claims caching with customizable date format
- feat(date): enhance date parsing with timezone support

## 2026-04

- feat: add scripts for training and excel generation
- feat(excel): add image extraction scripts
- feat(receipt): receipt/invoice processor

## 2026-03

- feat(transcript): speaker diarization and speech-to-text
- feat(crawler): async site crawling with 404 detection
- feat(pdf): auto rotate, extract images, compile PNG as PDF
- feat(image): remove background, scale, invert color, remove text
- feat(ocr): pytesseract OCR and CSV export
