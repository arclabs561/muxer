//! Deterministic hashing helpers for tie-breaks and “randomized” ordering.
//!
//! This module intentionally does **not** provide cryptographic guarantees; it is meant for
//! repeatable tie-breaking and stable pseudo-random ordering in routing policies.

/// Deterministic (non-crypto) stable hash used for “random” sampling / tie-breaking.
///
/// Implementation:
/// - FNV-1a over bytes (cheap, stable across platforms)
/// - SplitMix64 finalizer (improves bit diffusion / uniformity)
#[must_use]
pub fn stable_hash64(seed: u64, s: &str) -> u64 {
    let mut h: u64 = 14695981039346656037u64;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(1099511628211u64);
    }
    stable_hash64_u64(seed, h)
}

/// Deterministic (non-crypto) stable hash for a 64-bit value.
///
/// Useful for tie-breaks where the “key” is already numeric (e.g. arm index).
#[must_use]
pub fn stable_hash64_u64(seed: u64, x: u64) -> u64 {
    splitmix64(seed ^ x)
}

/// Deterministic u64-to-`[0,1)` mapping (53-bit mantissa precision).
///
/// Useful for seeded sampling from probability distributions without persisting RNG state.
#[cfg(feature = "stochastic")]
#[must_use]
pub(crate) fn u01_from_seed(seed: u64) -> f64 {
    let x = splitmix64(seed);
    let top = x >> 11; // 53 bits
    (top as f64) / ((1u64 << 53) as f64)
}

#[inline]
#[cfg_attr(not(feature = "stochastic"), allow(dead_code))]
pub(crate) fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
