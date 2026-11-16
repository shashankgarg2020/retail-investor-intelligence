FROM python:3.12-slim

# A non-root user (uid 1000), as most container hosts expect.
RUN useradd --create-home --uid 1000 app
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/.cache/huggingface \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed PATH=/home/app/.local/bin:$PATH

# The full image includes the FinBERT baseline. CPU-only torch keeps it several GB smaller
# than the default CUDA build. (Streamlit Community Cloud uses requirements.txt, without torch.)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt requirements-ml.txt ./
RUN pip install -r requirements-ml.txt

COPY --chown=app:app . .
RUN mkdir -p "$HF_HOME" "$FASTEMBED_CACHE_PATH" && chown -R app:app /app && chmod +x scripts/start_container.sh
USER app

# Download the models at build time, so the first refresh and first question do not wait.
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='ProsusAI/finbert'); \
from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

# 8501: dashboard (public). 8000: API (internal in the single-container setup).
EXPOSE 8501 8000
CMD ["./scripts/start_container.sh"]
