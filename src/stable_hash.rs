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
    splitmix64(seed ^ h)
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

