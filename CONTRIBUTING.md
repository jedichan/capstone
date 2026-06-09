# Contributing

This repository is a public source release for the Capstone surveillance video
query system. Keep changes focused and avoid committing generated artifacts.

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
npm --prefix frontend install
cp .env.example .env
```

Fill `.env` with local keys and endpoints. Never commit `.env` or real keys.

## Development

Run the API:

```bash
make dev-api
```

Run the UI:

```bash
make dev-ui
```

Run lightweight checks:

```bash
make test-smoke
```

Some tests and scripts require local videos, generated runtime databases, model
weights, GPU support, or hosted model API keys. Mark those as environment
dependent when reporting results.

## Git Hygiene

Do not commit:

- Reports, submitted coursework documents, PDFs, slides, or demo videos.
- Uploaded videos, runtime databases, model weights, Chroma stores, or SQLite
  artifacts.
- `.env`, API keys, tokens, local paths, or private credentials.
- `node_modules`, frontend build output, logs, or experiment outputs.

Prefer small commits that describe one change clearly.
