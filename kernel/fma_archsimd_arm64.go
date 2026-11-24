//go:build goexperiment.simd && arm64

package kernel

import "simd/archsimd"

// FmaArchSIMD computes dst[i] = a[i]*b[i] + c[i] using ARM64 NEON 128-bit vectors.
func FmaArchSIMD(dst, a, b, c []float32) {
	n := minLen(len(dst), len(a), len(b), len(c))
	if n == 0 {
		return
	}

	const lanes = 4
	i := 0
	for ; i+lanes <= n; i += lanes {
		va := archsimd.LoadFloat32x4(a[i : i+lanes])
		vb := archsimd.LoadFloat32x4(b[i : i+lanes])
		vc := archsimd.LoadFloat32x4(c[i : i+lanes])
		va.MulAdd(vb, vc).Store(dst[i : i+lanes])
	}

	if i < n {
		va, _ := archsimd.LoadFloat32x4Part(a[i:])
		vb, _ := archsimd.LoadFloat32x4Part(b[i:])
		vc, _ := archsimd.LoadFloat32x4Part(c[i:])
		va.MulAdd(vb, vc).StorePart(dst[i:])
	}
}