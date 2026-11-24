package kernel

/*
#cgo LDFLAGS: -L../rust-simd/target/release -lfma_rust_simd -lm
#include <stddef.h>

extern void fma_f32_slice(float* dst, const float* a, const float* b, const float* c, size_t len);
*/
import "C"
import "unsafe"

// FmaRustCgo calls the Rust FMA implementation across the cgo boundary.
func FmaRustCgo(dst, a, b, c []float32) {
	n := minLen(len(dst), len(a), len(b), len(c))
	if n == 0 {
		return
	}

	C.fma_f32_slice(
		(*C.float)(unsafe.Pointer(&dst[0])),
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&b[0])),
		(*C.float)(unsafe.Pointer(&c[0])),
		C.size_t(n),
	)
}