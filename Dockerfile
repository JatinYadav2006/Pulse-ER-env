FROM clashking9999/pulse-er-env:latest

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PULSE_INSTALL_DIR=/app/engine-build/install \
    PYTHONPATH=/workspace

WORKDIR /workspace

COPY . /workspace

RUN python -m pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e /workspace protobuf==5.29.2

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "pulse_physiology_env.server.app", "--port", "8000"]
