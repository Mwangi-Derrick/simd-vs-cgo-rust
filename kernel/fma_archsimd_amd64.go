//go:build goexperiment.simd && amd64

package kernel

import "simd/archsimd"

// FmaArchSIMD computes dst[i] = a[i]*b[i] + c[i] using AVX2 256-bit vectors when available.
func FmaArchSIMD(dst, a, b, c []float32) {
	n := minLen(len(dst), len(a), len(b), len(c))
	if n == 0 {
		return
	}
	if !archsimd.X86.AVX2() {
		FmaScalar(dst[:n], a[:n], b[:n], c[:n])
		return
	}

	const lanes = 8
	i := 0
	for ; i+lanes <= n; i += lanes {
		va := archsimd.LoadFloat32x8(a[i : i+lanes])
		vb := archsimd.LoadFloat32x8(b[i : i+lanes])
		vc := archsimd.LoadFloat32x8(c[i : i+lanes])
		va.MulAdd(vb, vc).Store(dst[i : i+lanes])
	}

	if i < n {
		va, _ := archsimd.LoadFloat32x8Part(a[i:])
		vb, _ := archsimd.LoadFloat32x8Part(b[i:])
		vc, _ := archsimd.LoadFloat32x8Part(c[i:])
		va.MulAdd(vb, vc).StorePart(dst[i:])
	}
}