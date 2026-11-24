package kernel

// FmaScalar computes dst[i] = a[i]*b[i] + c[i].
func FmaScalar(dst, a, b, c []float32) {
	n := minLen(len(dst), len(a), len(b), len(c))
	for i := 0; i < n; i++ {
		dst[i] = a[i]*b[i] + c[i]
	}
}

func minLen(a, b, c, d int) int {
	m := a
	if b < m {
		m = b
	}
	if c < m {
		m = c
	}
	if d < m {
		m = d
	}
	return m
}