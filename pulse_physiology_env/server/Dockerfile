FROM python:3.12-slim AS pulse-builder

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /build

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      clang \
      cmake \
      curl \
      git \
      libc++-dev \
      libc++abi-dev \
      libomp-dev \
      wget && \
    rm -rf /var/lib/apt/lists/*

COPY engine /source/engine

RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r /source/engine/requirements.txt

RUN cmake \
      -S /source/engine \
      -B /build/pulse \
      -DPulse_PYTHON_API=ON \
      -DPulse_JAVA_API=OFF \
      -DPulse_CSHARP_API=OFF \
      -DPulse_DOWNLOAD_BASELINES=OFF \
      -DCMAKE_INSTALL_PREFIX=/pulse \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_CXX_FLAGS=-stdlib=libc++ && \
    cmake --build /build/pulse --parallel 4 && \
    test -d /pulse/bin && \
    test -d /pulse/python


FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PULSE_INSTALL_DIR=/app/engine-build/install \
    PYTHONPATH=/app:/app/pulse_physiology_env

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      curl \
      libc++1 \
      libc++abi1 \
      libgomp1 \
      libssl-dev && \
    rm -rf /var/lib/apt/lists/*

COPY pulse_physiology_env/server/requirements.txt /app/requirements.txt
COPY engine/requirements.txt /app/engine-requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt -r /app/engine-requirements.txt protobuf==5.29.2

COPY pulse_physiology_env /app/pulse_physiology_env
COPY --from=pulse-builder /pulse /app/engine-build/install
COPY engine-build/install/bin/states /app/engine-build/install/bin/states

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

# This image builds a Linux Pulse install from source, so it can run on
# remote Linux workers such as Hugging Face Jobs without any host-mounted
# engine directory.
CMD ["python", "-m", "pulse_physiology_env.server.app", "--port", "8000"]
