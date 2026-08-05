# Real hipcc device-code compiler, no AMD GPU needed: HIP device-code
# compilation to a loadable code object never touches a driver or /dev/kfd,
# only the ROCm LLVM/HIP toolchain. See README.md for what this checks (and
# does not check).
FROM rocm/dev-ubuntu-22.04:6.2

COPY check-kernel.sh /usr/local/bin/check-kernel.sh
RUN chmod +x /usr/local/bin/check-kernel.sh

WORKDIR /work
ENTRYPOINT ["/usr/local/bin/check-kernel.sh"]
