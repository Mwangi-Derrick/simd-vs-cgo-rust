use core::slice;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{_mm256_fmadd_ps, _mm256_loadu_ps, _mm256_storeu_ps};

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{vfmaq_f32, vld1q_f32, vst1q_f32};

/// Computes `dst[i] = a[i] * b[i] + c[i]` using explicit SIMD intrinsics when available.
///
/// # Safety
/// `dst`, `a`, `b`, and `c` must each point to at least `len` initialized `f32` elements
/// and not overlap with `dst` in ways that violate aliasing rules.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fma_f32_slice(
    dst: *mut f32,
    a: *const f32,
    b: *const f32,
    c: *const f32,
    len: usize,
) {
    if len == 0 || dst.is_null() || a.is_null() || b.is_null() || c.is_null() {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("fma") {
        // SAFETY: Runtime check confirmed FMA (and AVX) support; caller guarantees valid slices.
        unsafe { fma_avx_fma_x86_64(dst, a, b, c, len) };
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is architectural on AArch64; caller guarantees valid slices.
        unsafe { fma_neon_aarch64(dst, a, b, c, len) };
        return;
    }

    // Fallback for targets without explicit intrinsic path above
    // SAFETY: Caller guarantees non-null pointers valid for `len` elements.
    let dst_s = unsafe { slice::from_raw_parts_mut(dst, len) };
    let a_s = unsafe { slice::from_raw_parts(a, len) };
    let b_s = unsafe { slice::from_raw_parts(b, len) };
    let c_s = unsafe { slice::from_raw_parts(c, len) };
    for (((d, &x), &y), &z) in dst_s
        .iter_mut()
        .zip(a_s.iter())
        .zip(b_s.iter())
        .zip(c_s.iter())
    {
        *d = x.mul_add(y, z);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
unsafe fn fma_avx_fma_x86_64(
    mut dst: *mut f32,
    mut a: *const f32,
    mut b: *const f32,
    mut c: *const f32,
    len: usize,
) {
    const LANES: usize = 8;
    let chunks = len / LANES;
    let rem = len % LANES;

    for _ in 0..chunks {
        // SAFETY: `chunks * 8 <= len` and pointers are valid for `len` elements.
        unsafe {
            let va = _mm256_loadu_ps(a);
            let vb = _mm256_loadu_ps(b);
            let vc = _mm256_loadu_ps(c);
            let out = _mm256_fmadd_ps(va, vb, vc);
            _mm256_storeu_ps(dst, out);
            a = a.add(LANES);
            b = b.add(LANES);
            c = c.add(LANES);
            dst = dst.add(LANES);
        }
    }

    // SAFETY: Remaining `rem < 8` elements are in-bounds.
    let dst_tail = unsafe { slice::from_raw_parts_mut(dst, rem) };
    let a_tail = unsafe { slice::from_raw_parts(a, rem) };
    let b_tail = unsafe { slice::from_raw_parts(b, rem) };
    let c_tail = unsafe { slice::from_raw_parts(c, rem) };
    for i in 0..rem {
        dst_tail[i] = a_tail[i].mul_add(b_tail[i], c_tail[i]);
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn fma_neon_aarch64(
    mut dst: *mut f32,
    mut a: *const f32,
    mut b: *const f32,
    mut c: *const f32,
    len: usize,
) {
    const LANES: usize = 4;
    let chunks = len / LANES;
    let rem = len % LANES;

    for _ in 0..chunks {
        // SAFETY: `chunks * 4 <= len` and pointers are valid for `len` elements.
        unsafe {
            let va = vld1q_f32(a);
            let vb = vld1q_f32(b);
            let vc = vld1q_f32(c);
            // Note: vfmaq_f32(c, a, b) computes c + a*b
            let out = vfmaq_f32(vc, va, vb);
            vst1q_f32(dst, out);
            a = a.add(LANES);
            b = b.add(LANES);
            c = c.add(LANES);
            dst = dst.add(LANES);
        }
    }

    // SAFETY: Remaining `rem < 4` elements are in-bounds.
    let dst_tail = unsafe { slice::from_raw_parts_mut(dst, rem) };
    let a_tail = unsafe { slice::from_raw_parts(a, rem) };
    let b_tail = unsafe { slice::from_raw_parts(b, rem) };
    let c_tail = unsafe { slice::from_raw_parts(c, rem) };
    for i in 0..rem {
        dst_tail[i] = a_tail[i].mul_add(b_tail[i], c_tail[i]);
    }
}