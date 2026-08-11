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
// `Copy`, like every real Bevy/glam math type it stands in for (`Vec2`/`Vec3`/...) -- needed
// to read it back out through `.pointee`'s `*PADDLE_SIZE` at a struct-literal field-value
// position (`Shape { pos: *PADDLE_SIZE }`) without moving out of the `'static` binding.
#[derive(Clone, Copy)]
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

// `Shape` stands in for a Bevy component holding a `Point2`-typed field
// (e.g. `Transform { translation: Vec2, .. }`) -- constructing one via a struct literal
// with `pos: PADDLE_SIZE` (the non-const `Point2.new(...)` promoted `let` below) is the
// struct-literal-field-VALUE position that a `static ... LazyLock<Point2>` cannot satisfy
// without an explicit `.pointee` deref -- unlike `PADDLE_SIZE.x`, which auto-derefs fine.
// See tests/cases/ext_const_promotion/src/main.br for the Boring-side exercise of both.
pub struct Shape {
    pub pos: Point2,
}

// `Color` mirrors `bevy_color::Color::srgb` -- genuinely `pub const fn` in its real
// defining crate (hand-verified against `crates/bevy_color/src/color.rs` in
// bevyengine/bevy; see `Transpiler::KNOWN_EXTERNAL_CONST_FNS` in src/transpiler/mod.rs).
// Exercises the `const` promotion path (not the `static ... LazyLock` fallback `Point2`
// above exercises) for a top-level `let` used at the exact struct-literal field-value
// position that motivated this whole fixture: `Sprite { color: PADDLE_COLOR, .. }`.
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl Color {
    pub const fn srgb(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }
}

// Stands in for Bevy's `Sprite { color: Color, .. }` -- the container whose field takes
// the external `Color` type directly.
pub struct Sprite {
    pub color: Color,
}

// `FontSize` mirrors `bevy_text::FontSize` (real shape:
// `bevy_text-0.19.0/src/text.rs:487`, `Px(f32) | Vw(f32) | Vh(f32) | VMin(f32) ...`) --
// an external *enum*, unlike `Color`/`Point2` above which are external structs. Its
// tuple-variant construction via Boring's dot-shorthand (`FontSize.Px(33.0)` →
// `FontSize::Px(33.0)`) is used by tests/cases/ext_enum_const_promotion/src/main.br to
// exercise `Transpiler::is_external_enum_variant_construction` (src/transpiler/mod.rs):
// unlike `Color::srgb` above, an enum tuple-variant construction is *always*
// const-evaluable in Rust regardless of which enum/variant it is, so it needs no
// `KNOWN_EXTERNAL_CONST_FNS`-style per-constructor hand-verification to promote to a
// plain `const` correctly.
#[derive(Clone, Copy)]
pub enum FontSize {
    Px(f32),
}

// Stands in for Bevy's `TextFont { font_size: FontSize, .. }` -- the struct-literal
// field-VALUE position that is the actual bug: a promoted `let` left as
// `static ... LazyLock<FontSize>` (instead of a plain `const`) fails to compile here
// with E0308 ("expected FontSize, found LazyLock<FontSize>"), the same failure mode
// `PADDLE_SIZE`/`PADDLE_COLOR` above hit before their own fix, just via a different
// initializer shape (enum tuple-variant construction, not an associated-function call).
pub struct TextLabel {
    pub font_size: FontSize,
}
