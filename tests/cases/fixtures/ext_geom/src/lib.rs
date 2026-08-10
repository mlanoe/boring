// Minimal external (non-Boring) crate used by the `ext_const_promotion` transpiler
// regression test (tests/cases/ext_const_promotion/) -- stands in for a real external
// crate like `bevy`/`glam`. `Point2` mirrors the shape of `bevy::math::Vec2` that
// motivated the bug this test guards: a plain struct with public fields, constructed via
// an associated function the Boring transpiler has no special knowledge of.
//
// `new` is deliberately NOT a `const fn` (the `Vec` round-trip forces a heap allocation,
// which a `const fn` can never do) -- this exercises the transpiler's `static ... LazyLock`
// fallback path for a top-level `let` whose external-type initializer isn't hand-verified
// as const-evaluable (see `Transpiler::KNOWN_EXTERNAL_CONST_FNS`'s doc in src/transpiler/mod.rs).
pub struct Point2 {
    pub x: f64,
    pub y: f64,
}

impl Point2 {
    pub fn new(x: f64, y: f64) -> Self {
        let coords = vec![x, y];
        Point2 { x: coords[0], y: coords[1] }
    }
}
