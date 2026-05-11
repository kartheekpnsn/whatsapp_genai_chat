.PHONY: dev backend frontend index

dev: ## Run backend and frontend concurrently
	@echo "Starting backend on :8003 and frontend on :5174 ..."
	@trap 'kill 0' INT; \
	uv run uvicorn whatsapp_genai_chat.api.main:app --port 8003 --reload & \
	cd frontend && npm run dev -- --port 5174; \
	wait

backend: ## Run FastAPI backend on port 8003
	uv run uvicorn whatsapp_genai_chat.api.main:app --port 8003 --reload

frontend: ## Run Vite.js frontend on port 5174
	cd frontend && npm run dev -- --port 5174

index: ## Build FAISS index. Usage: make index FILE=data/chat-sample.txt
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make index FILE=data/chat-sample.txt"; \
		exit 1; \
	fi
	uv run scripts/build_index.py $(FILE)
