# Capstone FastAPI Deploy Guide

## Goal

- Deploy the MEVID FastAPI service and the React Web UI from this repository.
- The current API service lives under `mevid_fastapi/` and defaults to port `8765`.
- The current frontend lives under `frontend/`, defaults to port `5173`, and forwards `/api` to `8765` via the Vite proxy.

## Prerequisites

- The repository has been cloned to the target directory.
- `conda` is installed and a usable `capstone` environment exists (or any Python 3.10/3.11 environment).
- The target machine has the models, databases, and API keys required by the project.
- For public access, a reverse proxy or port-exposure policy is ready.

## Recommended Approach

- Local development or single-machine validation:
  - Start the `mevid_fastapi` backend and the `frontend` separately.
- Long-running deployment:
  - Use `systemd` (Linux) or an equivalent process manager.

## Local Quick Start

Start the backend in the first terminal (run from the repository root):

```powershell
conda activate capstone
python -m uvicorn main:app --app-dir mevid_fastapi --host 127.0.0.1 --port 8765
```

Start the frontend in the second terminal:

```powershell
cd frontend
npm install
npm run dev -- --host 127.0.0.1
```

Default URL:

```text
http://127.0.0.1:5173/
```

## Manual / LAN Start

To expose the backend on the local network:

```powershell
python -m uvicorn main:app --app-dir mevid_fastapi --host 0.0.0.0 --port 8765
```

For LAN access to the frontend:

```powershell
npm run dev -- --host 0.0.0.0
```

## systemd Deployment (Linux)

For a long-running deployment, create a service that runs the same `uvicorn` command shown above, for example:

```ini
[Unit]
Description=Capstone MEVID FastAPI
After=network.target

[Service]
WorkingDirectory=/opt/capstone
ExecStart=/opt/conda/envs/capstone/bin/python -m uvicorn main:app --app-dir mevid_fastapi --host 0.0.0.0 --port 8765
Restart=always
EnvironmentFile=/opt/capstone/.env

[Install]
WantedBy=multi-user.target
```

Common commands:

```bash
sudo systemctl status capstone-fastapi.service
sudo systemctl restart capstone-fastapi.service
sudo journalctl -u capstone-fastapi.service -f
```

## Required Environment Variables

Copy `.env.example` to `.env` and fill in real values. Common variables:

- `DASHSCOPE_API_KEY`
- `DASHSCOPE_URL`
- `OPENAI_API_KEY` (optional)
- `OPENAI_BASE_URL` (optional)
- `AGENT_CHAT_MODEL`
- `AGENT_ENABLE_RERANK`
- `FASTAPI_MAX_UPLOAD_MB`
- `HOST`
- `PORT`

If video refinement or hosted-model capabilities are used, the target environment must provide the corresponding API key.

## Verification Steps

After startup, check the health endpoint first:

```bash
curl http://127.0.0.1:8765/healthz
```

Then open the UI in a browser:

```text
http://127.0.0.1:5173/
```

Recommended verification flow:

1. Open the page and confirm the frontend is reachable.
2. On the Query Agent page, select a prepared database (for example `mevid`) or upload a video to build a new one.
3. Run one query.
4. Upload one or more videos.
5. Confirm the system automatically switches to the new database.
6. Run another query and confirm the new database is usable.

## Troubleshooting

- Page does not open:
  - Check `HOST`, `PORT`, firewall, and reverse-proxy configuration.
- Service starts but queries fail:
  - Check the API key, database paths, model dependencies, and the Python environment.
- Upload fails:
  - Check the file format, the per-file size limit, and the backend logs.
- Upload succeeds but the new database is not listed:
  - Confirm `mevid_fastapi/data` is correctly persisted.
  - Check the backend logs to confirm the task completed without errors.
- `uvicorn` fails to start:
  - Confirm the directory entry point is correct:
  - `uvicorn main:app --app-dir mevid_fastapi --port 8765`

## Related Files

- `mevid_fastapi/main.py`
- `mevid_fastapi/API_REFERENCE.md`
- `frontend/package.json`
- `frontend/vite.config.js`
