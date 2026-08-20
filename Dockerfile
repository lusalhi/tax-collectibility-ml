FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./
COPY src/ ./src/
COPY models/ ./models/
COPY .streamlit/ ./.streamlit/

# Non-root user
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /home/appuser/.streamlit \
    && chown -R appuser:appuser /app /home/appuser
USER appuser

# Dalam container, app listen 0.0.0.0; akses dari luar hanya lewat reverse proxy (Caddy).
# Port TIDAK di-publish ke host oleh compose untuk service ini.
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"

CMD ["streamlit", "run", "app.py", "--server.headless", "true", "--server.address", "0.0.0.0", "--server.port", "8501"]
