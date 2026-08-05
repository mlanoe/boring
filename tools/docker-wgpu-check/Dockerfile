# Two checks in one image, no real GPU needed:
#   1. naga (the same WGSL front-end wgpu itself uses) parses/type-checks
#      shaders/*.wgsl -- no Vulkan, no GPU, no device at all involved.
#   2. (opt-in, MODE=run) actually builds and runs the generated project's
#      compute dispatch through wgpu's Vulkan backend against Mesa's
#      lavapipe -- a real, if CPU-slow, software Vulkan device. Genuine
#      execution, not just a static check. See README.md.
FROM rust:1-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential pkg-config ca-certificates \
        mesa-vulkan-drivers libvulkan1 vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

RUN cargo install naga-cli --locked

COPY check-shader.sh /usr/local/bin/check-shader.sh
RUN chmod +x /usr/local/bin/check-shader.sh

WORKDIR /work
ENTRYPOINT ["/usr/local/bin/check-shader.sh"]
