FROM python:3.12-slim AS builder
WORKDIR /app

# Install git and C++ build tools required to clone and build hisss
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

FROM python:3.12-slim
WORKDIR /app

# Copy the compiled packages (including hisss) into runtime
COPY --from=builder /install /usr/local
COPY . .

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
RUN useradd -m snake
USER snake

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["python", "main.py"]