# API server only. Training needs PyTorch and runs elsewhere; the image just loads ONNX.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    C2T_MODELS_DIR=/models \
    C2T_DEFAULT_MODEL=current \
    C2T_DB_PATH=/data/captcha2text.db

WORKDIR /app
COPY pyproject.toml README.md ./
COPY captcha2text ./captcha2text
RUN pip install --no-cache-dir . \
 && useradd --system --uid 10001 --home /app c2t \
 && mkdir -p /data && chown c2t /data

USER c2t
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/v1/health', timeout=2).status == 200 else 1)"

# One process: requests run on threads and ONNX Runtime uses several cores per image,
# and the in-memory rate limiter needs every request to go through the same process.
CMD ["uvicorn", "captcha2text.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-server-header"]
