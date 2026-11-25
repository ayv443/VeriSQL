# VeriSQL API.
#   docker build -t verisql .                       # small image, replay engine (no model, no GPU)
#   docker run --rm -p 8000:8000 verisql            # then open http://localhost:8000/docs
#
# With the real model on CPU (about 3 GB more), adapters pulled from the Hugging Face Hub at startup:
#   docker build -t verisql:model --build-arg WITH_MODEL=1 .
#   docker run --rm -p 8000:8000 -e VERISQL_ENGINE=live \
#       -e VERISQL_ADAPTER=ayv443/verisql-adapters -e VERISQL_SUBFOLDER=mt-grpo-step150 verisql:model
FROM python:3.11-slim

ARG WITH_MODEL=0
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/.cache/hf

WORKDIR /app
COPY requirements-service.txt requirements-model.txt ./
RUN pip install -r requirements-service.txt && \
    if [ "$WITH_MODEL" = "1" ]; then \
      pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu && \
      pip install -r requirements-model.txt; \
    fi

# only what the service needs: the loop, the shared helpers, the saved runs for replay, the eval databases
COPY text2sql-rl/common.py text2sql-rl/reward.py text2sql-rl/
COPY extension/multiturn_eval.py extension/
COPY extension/findings/multiturn/mt-grpo extension/findings/multiturn/mt-grpo
COPY verisql/core.py verisql/
COPY service service
COPY eval eval

RUN useradd --create-home verisql && mkdir -p /app/.cache && chown -R verisql /app/.cache
USER verisql
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000"]
