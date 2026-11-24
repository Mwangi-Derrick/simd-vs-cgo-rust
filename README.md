# Go 1.27 SIMD vs. Rust (`cgo`) SIMD Benchmark

A multi-architecture benchmark and verification harness comparing **Go 1.27's experimental SIMD support** (`simd` and `simd/archsimd`) against a **Rust `staticlib` called over `cgo`** using explicit `core::arch` SIMD intrinsics (`AVX2` + `FMA` on `x86_64`, `NEON` `vfmaq_f32` on `aarch64`).

The probe kernel is a `float32` fused multiply-add sweep:

`dst[i] = a[i] * b[i] + c[i]`

across varying element counts `N` to isolate:

- **`cgo` boundary overhead** vs. SIMD compute amortization at small vs. large buffer sizes.
- **Portable `simd` vs. architecture-specific `simd/archsimd`** (`Float32x8` on `amd64`, `Float32x4` on `arm64`).
- **Effective memory bandwidth (`GB/s`)** (`16 * N` bytes/op across three `float32` reads and one write) alongside median latency (`ns/op`).
- **Compiler codegen verification** in CI via Rust assembly inspection and Go test-binary disassembly checks.

---

## Project Layout

```text
simd-vs-cgo-rust/
├── go.mod
├── kernel/
│   ├── fma_scalar.go          # Scalar reference implementation & shared helpers
│   ├── fma_go_simd.go         # Go 1.27 portable `simd` implementation (`goexperiment.simd`)
│   ├── fma_archsimd_amd64.go  # AVX2 (`Float32x8`) fast path with runtime feature check
│   ├── fma_archsimd_arm64.go  # NEON (`Float32x4`) fast path for AArch64
│   ├── fma_archsimd_other.go  # Scalar fallback on non-amd64/arm64 targets
│   ├── fma_cgo_rust.go        # Go `cgo` wrapper invoking `fma_f32_slice`
│   └── fma_bench_test.go      # Correctness tests & buffer-size-swept benchmarks
├── rust-simd/
│   ├── Cargo.toml             # `staticlib` crate exporting `extern "C"` entry point
│   └── src/lib.rs             # Explicit `core::arch` FMA/NEON intrinsics + fallback
└── .github/
    └── workflows/
        └── bench.yml          # Multi-arch CI (amd64 & arm64), asm gates, benchstat & summary
```

## Prerequisites

- Go 1.27+ with `GOEXPERIMENT=simd`
- Rust toolchain (`cargo` / `rustc`)
- C compiler (`gcc` or `clang`) for cgo
- Optional: `benchstat` (`go install golang.org/x/perf/cmd/benchstat@latest`)

## Running Locally

Build the Rust static library:

```bash
cd rust-simd
cargo build --release
cd ..
```


## Run correctness tests:

```bash
GOEXPERIMENT=simd go test -v ./kernel/...
Run benchmarks:
```

```bash
GOEXPERIMENT=simd go test -bench=BenchmarkFma -benchmem -count=6 ./kernel/...
Inspecting Generated Assembly
Rust (rustc emitted .s)
```

```bash
cd rust-simd
RUSTFLAGS="--emit asm" cargo rustc --release -- --emit asm
find target/release/deps -name "*.s" -exec grep -n -C 10 "fma_avx_fma_x86_64\|fma_neon_aarch64" {} +
Go (go tool objdump & native objdump)
```

```bash
GOEXPERIMENT=simd go test -c -o /tmp/kernel.test ./kernel
go tool objdump -s 'kernel\.(FmaGoSIMD|FmaArchSIMD)' /tmp/kernel.test
```

### CI Pipeline (.github/workflows/bench.yml)
Runs across amd64 and arm64 runners and:

Builds the Rust staticlib and emits assembly, failing CI if expected vector instructions (vfmadd on x86_64, fmla on aarch64) are absent.

Compiles the Go test binary once, dumps FmaGoSIMD and FmaArchSIMD disassembly, and verifies expected SIMD/FMA instructions.

Executes correctness tests and grouped benchmarks (-count=10) for GoSIMD, GoArchSIMD, and RustCgo.

Generates benchstat comparisons and a markdown summary (crossover-${{ matrix.arch }}.txt, also appended to $GITHUB_STEP_SUMMARY) reporting the smallest buffer size N where RustCgo wins, per-size median latency (ns/op), and effective bandwidth (GB/s).