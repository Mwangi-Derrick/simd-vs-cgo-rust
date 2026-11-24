//go:build goexperiment.simd

package kernel

import "simd"

// FmaGoSIMD computes dst[i] = a[i]*b[i] + c[i] using Go 1.27's portable simd package.
func FmaGoSIMD(dst, a, b, c []float32) {
	n := minLen(len(dst), len(a), len(b), len(c))
	if n == 0 {
		return
	}

	var zeroV simd.Float32s
	lanes := zeroV.Len()

	i := 0
	for ; i+lanes <= n; i += lanes {
		va := simd.LoadFloat32s(a[i : i+lanes])
		vb := simd.LoadFloat32s(b[i : i+lanes])
		vc := simd.LoadFloat32s(c[i : i+lanes])
		va.MulAdd(vb, vc).Store(dst[i : i+lanes])
	}

	if i < n {
		va, _ := simd.LoadFloat32sPart(a[i:])
		vb, _ := simd.LoadFloat32sPart(b[i:])
		vc, _ := simd.LoadFloat32sPart(c[i:])
		va.MulAdd(vb, vc).StorePart(dst[i:])
	}
}