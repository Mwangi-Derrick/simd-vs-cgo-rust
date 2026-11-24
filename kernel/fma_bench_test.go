//go:build goexperiment.simd

package kernel

import (
	"math"
	"testing"
)

var sink float32

func makeBufs(n int) (dst, a, b, c []float32) {
	dst = make([]float32, n)
	a = make([]float32, n)
	b = make([]float32, n)
	c = make([]float32, n)
	for i := 0; i < n; i++ {
		a[i] = float32(i%97)*0.125 + 1.0
		b[i] = float32(i%53)*0.25 - 0.5
		c[i] = float32(i%31) * 0.5
	}
	return dst, a, b, c
}

func TestFmaCorrectness(t *testing.T) {
	sizes := []int{0, 1, 3, 4, 7, 8, 15, 16, 17, 64, 257, 4096}
	for _, n := range sizes {
		ref, a, b, c := makeBufs(n)
		FmaScalar(ref, a, b, c)

		gotGo := make([]float32, n)
		FmaGoSIMD(gotGo, a, b, c)

		gotRust := make([]float32, n)
		FmaRustCgo(gotRust, a, b, c)

		for i := 0; i < n; i++ {
			if diff := math.Abs(float64(ref[i] - gotGo[i])); diff > 1e-4 {
				t.Fatalf("n=%d i=%d GoSIMD mismatch: want %v got %v", n, i, ref[i], gotGo[i])
			}
			if diff := math.Abs(float64(ref[i] - gotRust[i])); diff > 1e-4 {
				t.Fatalf("n=%d i=%d RustCgo mismatch: want %v got %v", n, i, ref[i], gotRust[i])
			}
		}
	}
}

func runBench(b *testing.B, n int, fn func(dst, a, b, c []float32)) {
	dst, a, bBuf, c := makeBufs(n)
	b.SetBytes(int64(n * 4 * 4)) // 3 reads + 1 write of float32
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fn(dst, a, bBuf, c)
	}
	if n > 0 {
		sink = dst[n-1]
	}
}

func BenchmarkFma(b *testing.B) {
	sizes := []int{64, 1024, 65536, 1048576}
	for _, n := range sizes {
		b.Run("Scalar/N="+itoa(n), func(b *testing.B) { runBench(b, n, FmaScalar) })
		b.Run("GoSIMD/N="+itoa(n), func(b *testing.B) { runBench(b, n, FmaGoSIMD) })
		b.Run("RustCgo/N="+itoa(n), func(b *testing.B) { runBench(b, n, FmaRustCgo) })
	}
}

func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for v > 0 {
		i--
		buf[i] = byte('0' + v%10)
		v /= 10
	}
	return string(buf[i:])
}