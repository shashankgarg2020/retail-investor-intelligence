#!/usr/bin/env sh
# One container, two processes: the API on an internal port, the dashboard on the public one.
# Used by `docker run` of the image and container hosts such as Railway, Render or Fly.io (they set PORT).
set -e

PORT="${PORT:-8501}"
export API_BASE_URL="${API_BASE_URL:-http://127.0.0.1:8000}"

uvicorn src.api.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!

# Wait for the API before the dashboard starts taking visitors.
for _ in $(seq 1 60); do
  python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" \
    >/dev/null 2>&1 && break
  sleep 1
done

# If the API dies, stop the container so the platform restarts it.
( wait "$API_PID"; echo "API process exited"; kill 0 ) &

exec streamlit run src/frontend/app.py --server.port "$PORT" --server.address 0.0.0.0
