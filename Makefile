PYTHON ?= python3
HOST ?= 127.0.0.1
PORT ?= 8765
UI_HOST ?= 127.0.0.1

.PHONY: install-backend install-frontend dev-api dev-ui dev test-smoke build-ui clean

install-backend:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

install-frontend:
	npm --prefix frontend install

dev-api:
	$(PYTHON) -m uvicorn main:app --app-dir mevid_fastapi --host $(HOST) --port $(PORT)

dev-ui:
	npm --prefix frontend run dev -- --host $(UI_HOST)

dev:
	./quick_start.sh

test-smoke:
	$(PYTHON) -m compileall -q agent video mevid_fastapi scripts tests
	npm --prefix frontend run build

build-ui:
	npm --prefix frontend run build

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
