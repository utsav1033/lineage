FROM python:3.12-slim

# Spaces run containers as user 1000.
RUN useradd -m -u 1000 user
WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
USER user

ENV LINEAGE_CACHE_DIR=/tmp/lineage-cache \
    PYTHONUNBUFFERED=1
EXPOSE 7860
# Hosts like Render pass the port in $PORT.
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-7860} --proxy-headers --forwarded-allow-ips '*'"]
