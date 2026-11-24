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

For `benchstat` comparisons, install the tool and save each implementation's
samples to a separate file:

```bash
go install golang.org/x/perf/cmd/benchstat@latest
GOEXPERIMENT=simd go test -bench='BenchmarkFma/GoSIMD/' -benchmem -count=10 > go_simd.txt
GOEXPERIMENT=simd go test -bench='BenchmarkFma/GoArchSIMD/' -benchmem -count=10 > go_archsimd.txt
GOEXPERIMENT=simd go test -bench='BenchmarkFma/RustCgo/' -benchmem -count=10 > rust_cgo.txt
benchstat go_archsimd.txt rust_cgo.txt
benchstat go_simd.txt go_archsimd.txt
```

## Inspect Generated Assembly

From the repository root, emit Rust assembly and find the generated files:

```bash
cargo rustc --manifest-path rust-simd/Cargo.toml --release -- --emit=asm
find rust-simd/target/release -name '*.s' -print
grep -Ei -n -C 5 'vfmadd|fmla' rust-simd/target/release/build/fma_rust_simd/*/out/*.s
```

Inspect Go code generation from `kernel`:

```bash
GOEXPERIMENT=simd go test -c -o /tmp/kernel.test
go tool objdump -s 'simdprobe\.Fma(GoSIMD|ArchSIMD)' /tmp/kernel.test
```
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
cargo build --release --manifest-path rust-simd/Cargo.toml
```


The Go module is in `kernel`, so enter that directory before running Go commands:

```bash
cd kernel
GOEXPERIMENT=simd go test -v ./...
```

Run the grouped benchmarks:

```bash
GOEXPERIMENT=simd go test -bench=BenchmarkFma -benchmem -count=10
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

### CI Pipeline

CI runs on amd64 and arm64 runners. It builds the Rust static library, checks
the generated Rust and Go assembly for architecture-specific FMA instructions,
runs correctness tests, and collects ten samples for each benchmark
implementation. It compares GoArchSIMD against RustCgo and portable GoSIMD
against GoArchSIMD with `benchstat`, appends the comparisons to the Actions
summary, and uploads raw benchmark and assembly outputs as artifacts.