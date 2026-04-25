FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PULSE_INSTALL_DIR=/app/engine-build/install \
    PYTHONPATH=/app:/app/pulse_physiology_env

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential cmake libssl-dev curl && \
    rm -rf /var/lib/apt/lists/*

# Build this image from the repository root:
# docker build -f pulse_physiology_env/server/Dockerfile -t pulse-er .
COPY pulse_physiology_env/server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY pulse_physiology_env /app/pulse_physiology_env

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Mount engine-build from the host at runtime:
# docker run -v ${PWD}/engine-build:/app/engine-build -p 8000:8000 pulse-er
CMD ["python", "-m", "pulse_physiology_env.server.app", "--port", "8000"]
