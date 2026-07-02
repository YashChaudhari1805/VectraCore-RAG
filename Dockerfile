# ── Stage 1: Build the React Frontend ─────────────────────────────────────────
FROM node:20-slim AS frontend-builder

WORKDIR /app/frontend
# Copy package files and install dependencies
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install

# Copy the rest of the frontend code and build it
COPY frontend/ ./
# This will output to /app/dashboard based on our vite.config.ts
RUN npm run build


# ── Stage 2: Build Python Dependencies ────────────────────────────────────────
FROM python:3.11-slim AS backend-builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 3: Production Runtime ───────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed Python packages from Stage 2
COPY --from=backend-builder /install /usr/local

# Copy the backend code
COPY backend/ ./backend/

# Copy the compiled React dashboard from Stage 1
COPY --from=frontend-builder /app/frontend/dist ./dashboard/

# Prepare memory directory and non-root user for security
RUN mkdir -p data/memory
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Healthcheck ensures Render knows when the app is fully awake
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')"

# Launch the new backend entrypoint
CMD ["python", "backend/main.py"]