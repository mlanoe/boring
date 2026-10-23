// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Single-core micro-benchmark behind the CPU quantized-linear evaluation in docs/simd-backend.md:
// Q8_0, one activation row against 11008 columns of 2048 weights.
//
//   rustc --edition 2021 -C opt-level=3 -C target-cpu=native tools/simd_quant_microbench.rs -o /tmp/microbench && /tmp/microbench
//
// Variants: the original dequantize-then-dot order, a simple fused float loop, 16 float lanes kept
// across blocks (one horizontal sum per column), and an int8-activation NEON kernel built from
// stable intrinsics (`vdotq_s32` is still unstable in Rust, hence vmull_s8 + vpadalq_s16).
use std::time::Instant;
const K: usize = 2048;
const N: usize = 11008;
const BLK: usize = 34;

fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0f32 } else { -1.0 };
    let e = ((bits >> 10) & 0x1f) as i32; let f = (bits & 0x3ff) as f32;
    if e == 0 { sign * f / 16777216.0 } else { sign * (1.0 + f / 1024.0) * 2f32.powi(e - 15) }
}

#[inline(never)]
fn current(w: &[u8], a: &[f32], out: &mut [f32]) {
    // dequantize a block to f32, then 16-lane dot (what the SIMD target does today)
    let blocks = K / 32;
    for col in 0..N {
        let mut sum = 0f32;
        for b in 0..blocks {
            let blk = &w[(col * blocks + b) * BLK..][..BLK];
            let scale = f16_to_f32(u16::from(blk[0]) | (u16::from(blk[1]) << 8));
            let mut tmp = [0f32; 32];
            for i in 0..32 { tmp[i] = scale * (blk[2 + i] as i8 as f32); }
            let x = &a[b * 32..b * 32 + 32];
            let mut lanes = [0f32; 16];
            for i in 0..32 { lanes[i % 16] += tmp[i] * x[i]; }
            let mut s = 0f32; for l in lanes { s += l; }
            sum += s;
        }
        out[col] = sum;
    }
}

#[inline(never)]
fn fused_float(w: &[u8], a: &[f32], out: &mut [f32]) {
    // no intermediate array: convert i8 -> f32 and multiply-accumulate, scale applied once per block
    let blocks = K / 32;
    for col in 0..N {
        let mut sum = 0f32;
        for b in 0..blocks {
            let blk = &w[(col * blocks + b) * BLK..][..BLK];
            let scale = f16_to_f32(u16::from(blk[0]) | (u16::from(blk[1]) << 8));
            let x = &a[b * 32..b * 32 + 32];
            let mut acc = [0f32; 8];
            for i in 0..32 { acc[i % 8] += (blk[2 + i] as i8 as f32) * x[i]; }
            let mut s = 0f32; for l in acc { s += l; }
            sum += scale * s;
        }
        out[col] = sum;
    }
}

#[inline(never)]
fn lanes_across_blocks(w: &[u8], a: &[f32], out: &mut [f32]) {
    // block dequantizer (as shipped) + 16 persistent lanes, one horizontal reduction per column
    let blocks = K / 32;
    for col in 0..N {
        let mut acc = [0f32; 16];
        for b in 0..blocks {
            let blk = &w[(col * blocks + b) * BLK..][..BLK];
            let blk: &[u8; BLK] = blk.try_into().unwrap();
            let scale = f16_to_f32(u16::from(blk[0]) | (u16::from(blk[1]) << 8));
            let mut tmp = [0f32; 32];
            for i in 0..32 { tmp[i] = scale * (blk[2 + i] as i8 as f32); }
            let x = &a[b * 32..b * 32 + 32];
            for i in 0..32 { acc[i % 16] = x[i].mul_add(tmp[i], acc[i % 16]); }
        }
        let mut s = 0f32; for l in acc { s += l; }
        out[col] = s;
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn dot_neon(w: &[u8], qa: &[i8], sa: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;
    let blocks = K / 32;
    for col in 0..N {
        let mut sum = 0f32;
        for b in 0..blocks {
            let blk = &w[(col * blocks + b) * BLK..][..BLK];
            let scale = f16_to_f32(u16::from(blk[0]) | (u16::from(blk[1]) << 8));
            let wp = blk.as_ptr().add(2) as *const i8;
            let ap = qa.as_ptr().add(b * 32);
            let mut acc = vdupq_n_s32(0);
            for half in 0..2 {
                let (x, y) = (vld1q_s8(wp.add(half * 16)), vld1q_s8(ap.add(half * 16)));
                acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(x), vget_low_s8(y)));
                acc = vpadalq_s16(acc, vmull_high_s8(x, y));
            }
            sum += scale * sa[b] * (vaddvq_s32(acc) as f32);
        }
        out[col] = sum;
    }
}

fn main() {
    let blocks = K / 32;
    let mut w = vec![0u8; N * blocks * BLK];
    for (i, v) in w.iter_mut().enumerate() {
        let pos = i % BLK;
        *v = if pos == 0 { 0x00 } else if pos == 1 { 0x2c } else { ((i * 17 + 29) % 256) as u8 };
    }
    let a: Vec<f32> = (0..K).map(|i| ((i % 17) as f32) / 7.0 - 1.0).collect();
    // int8 activations (llama.cpp style): one scale per 32-element block
    let mut qa = vec![0i8; K]; let mut sa = vec![0f32; blocks];
    for b in 0..blocks {
        let m = a[b * 32..b * 32 + 32].iter().fold(0f32, |m, v| m.max(v.abs()));
        let s = m / 127.0; sa[b] = s;
        for i in 0..32 { qa[b * 32 + i] = (a[b * 32 + i] / s).round() as i8; }
    }
    let mut o1 = vec![0f32; N]; let mut o2 = vec![0f32; N]; let mut o3 = vec![0f32; N]; let mut o4 = vec![0f32; N];
    let bytes = (w.len()) as f64;
    let time = |name: &str, f: &mut dyn FnMut()| {
        f(); // warm
        let mut best = f64::MAX;
        for _ in 0..7 { let t = Instant::now(); f(); best = best.min(t.elapsed().as_secs_f64()); }
        println!("{name:<28} {:>7.3} ms/call   {:>5.1} GB/s (one core)", best * 1e3, bytes / best / 1e9);
    };
    time("current (dequant + dot)", &mut || current(&w, &a, &mut o1));
    time("fused float", &mut || fused_float(&w, &a, &mut o2));
    time("lanes across blocks", &mut || lanes_across_blocks(&w, &a, &mut o4));
    #[cfg(target_arch = "aarch64")]
    time("NEON int8 (vmull+vpadal)", &mut || unsafe { dot_neon(&w, &qa, &sa, &mut o3) });
    let err = |x: &[f32], y: &[f32]| x.iter().zip(y).fold(0f32, |m, (p, q)| m.max((p - q).abs())) / y.iter().fold(0f32, |m, v| m.max(v.abs()));
    println!("max rel diff vs current: fused {:.2e}, lanes {:.2e}, int8-dot {:.2e}", err(&o2, &o1), err(&o4, &o1), err(&o3, &o1));
}
