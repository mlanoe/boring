# Real nvcc/ptxas, no GPU needed: device-code compilation never touches a
# driver or a device, only the toolkit. See README.md for what this checks
# (and does not check).
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

COPY check-kernel.sh /usr/local/bin/check-kernel.sh
RUN chmod +x /usr/local/bin/check-kernel.sh

WORKDIR /work
ENTRYPOINT ["/usr/local/bin/check-kernel.sh"]
