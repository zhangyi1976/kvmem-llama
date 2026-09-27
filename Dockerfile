# KVMem + llama.cpp — CUDA 12.9 build with sm_70 (V100 / Volta) support.
# CUDA 13 dropped sm_70, so the V100-capable line is CUDA 12.9 (v0.16.0-rc3 release).
#
# Build args (all optional):
#   CUDA_IMAGE           CUDA devel image for the build stage
#   CUDA_RUNTIME_IMAGE   CUDA runtime image for the final stage (keep same version)
#   CUDA_ARCH            CMAKE_CUDA_ARCHITECTURES (default 70-real = V100)

ARG CUDA_IMAGE=nvidia/cuda:12.9.2-devel-ubuntu24.04
ARG CUDA_RUNTIME_IMAGE=nvidia/cuda:12.9.2-runtime-ubuntu24.04
FROM ${CUDA_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake g++ make python3 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Node.js 22 for building the browser UI (served by the server at /).
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .
# The build context must contain the pinned llama.cpp submodule sources.
RUN test -f llama.cpp/CMakeLists.txt || { echo "llama.cpp submodule missing from build context" >&2; exit 1; }

RUN bash scripts/apply-patches.sh
RUN python3 scripts/build-webui.py

ARG CUDA_ARCH=70-real
RUN cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
        -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=/src/build/bin \
        -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=/src/build/bin \
        # --allow-shlib-undefined: libggml-cuda.so needs the CUDA driver API
        # (libcuda.so.1) which only exists on a real GPU host; resolve at
        # runtime. Same fix as llama.cpp/.devops/cuda.Dockerfile.
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined -Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
        -DGGML_CUDA=ON \
        -DGGML_CUDA_FA_ALL_QUANTS=ON \
        -DGGML_NATIVE=OFF \
        -DKVMEM_BUILD_LLAMA=ON \
        -DLLAMA_KVMEM=ON \
        -DLLAMA_KVMEM_ROOT=/src \
    && cmake --build build --parallel "$(nproc)"

# ---- runtime ----
FROM ${CUDA_RUNTIME_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Keep the build/ layout: the server auto-serves ../share/kvmem/ui relative to
# the binary (tools/kvmem-webui.h).
COPY --from=builder /src/build/bin /opt/kvmem/bin
COPY --from=builder /src/build/share /opt/kvmem/share

ENV LD_LIBRARY_PATH=/opt/kvmem/bin:/usr/local/cuda/lib64
EXPOSE 18200
WORKDIR /models
ENTRYPOINT ["/opt/kvmem/bin/llama-kvmem-server"]
CMD ["--help"]
