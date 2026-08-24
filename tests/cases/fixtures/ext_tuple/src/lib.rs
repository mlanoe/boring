// Minimal external (non-Boring) crate used by the `ext_tuple_construct` transpiler
// regression test (tests/cases/ext_tuple_construct/) -- stands in for bevy's real
// `Mesh2d`/`MeshMaterial2d` (bevy_mesh 0.19 / bevy_sprite_render 0.19), which motivated
// this test: plain tuple structs with a public field and deliberately NO inherent
// `new()`. See `Transpiler::KNOWN_EXTERNAL_TUPLE_STRUCTS`'s doc in src/transpiler/mod.rs
// for the bug this guards against -- Boring's bare `Type(args)` constructor syntax used
// to unconditionally rewrite to `Type::new(args)`, which doesn't compile for a type like
// this (E0599, no `new` found).
pub struct Mesh2d(pub i64);

pub struct MeshMaterial2d<M>(pub M);

// Stands in for bevy_text 0.19's real `TextColor` (bevy_text-0.19.0/src/text.rs:1066),
// the motivating case for `Transpiler::KNOWN_EXTERNAL_TUPLE_STRUCTS`'s `TextColor` entry
// -- see tests/cases/text_color_construct/. Unlike `Mesh2d`/`MeshMaterial2d` above, the
// real `TextColor` DOES have an inherent `impl` block (associated consts `BLACK`/`WHITE`,
// mirrored below by `BLACK`) -- it's still not a constructor, so this deliberately keeps
// the trap that made the bug easy to miss: "has *an* inherent impl" is not the same
// signal as "has an inherent `new()`", and only the latter makes `Type::new(args)` valid.
pub struct TextColor(pub i64);

impl TextColor {
    pub const BLACK: TextColor = TextColor(0);
}

// Stands in for bevy_camera 0.19's real `ClearColor`
// (bevy_camera-0.19.0/src/clear_color.rs:55), the motivating case for
// `Transpiler::KNOWN_EXTERNAL_TUPLE_STRUCTS`'s `ClearColor` entry -- see
// tests/cases/clear_color_construct/. Unlike `TextColor` above, the real `ClearColor`
// has no associated consts either -- only a `Default` impl (still not a constructor),
// mirrored below so the fixture matches the real type's shape as closely as `Mesh2d`/
// `MeshMaterial2d` do.
pub struct ClearColor(pub i64);

impl Default for ClearColor {
    fn default() -> Self {
        ClearColor(0)
    }
}

// Deliberately NOT in `Transpiler::KNOWN_EXTERNAL_TUPLE_STRUCTS` -- stands in for a
// third-party tuple struct a project must whitelist itself via its own `boring.toml`
// `[external_types]` `tuple_structs = [...]` entry (see
// `tests/emit_rust_explicit_path.rs`), as opposed to `Mesh2d`/`TextColor`/`ClearColor`
// above, which the compiler already knows about built-in. No inherent `new()`, same
// shape as the others, so a project that forgot to apply its `[external_types]`
// supplement would wrongly rewrite `ExtOpaque(5)` to the nonexistent `ExtOpaque::new(5)`
// (E0599).
pub struct ExtOpaque(pub i64);
