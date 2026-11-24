//go:build goexperiment.simd && !amd64 && !arm64

package kernel

// FmaArchSIMD falls back to scalar on architectures without a dedicated archsimd fast path.
func FmaArchSIMD(dst, a, b, c []float32) {
	FmaScalar(dst, a, b, c)
}