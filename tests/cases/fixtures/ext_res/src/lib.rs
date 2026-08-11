// Minimal external (non-Boring) crate used by the `ext_res_field` transpiler regression
// test (tests/cases/ext_res_field/) -- stands in for real Bevy's `Res<T>`/`ResMut<T>`
// system-param types (bevy_ecs::system::{Res, ResMut}), which motivated this test: both
// are plain `Deref`/`DerefMut`-transparent wrappers around a real `T`, so a field access
// on the wrapper (`res.count`) is, at the Rust level, a field access on `T` through
// auto-deref -- exactly the property Boring's own field-resolution codegen must see
// through (see `TRANSPARENT_WRAPPER_GENERICS` in src/transpiler/emit_methods.rs).
//
// Real bevy's `Res<'w, T>` borrows (`&'w T`); this stand-in owns its value directly to
// keep the fixture (and the Boring source that constructs one) trivial -- the borrow-vs-
// own distinction plays no part in the bug being guarded against, only the Deref chain
// does. `Res::new`/`ResMut::new` exist so bare `Res(v)`/`ResMut(v)` construction from
// Boring source falls through to the transpiler's default `Type::new(args)` external-
// constructor fallback (see `emit_constructor_inner` in src/transpiler/emit_expr.rs) --
// no allowlist entry needed, unlike `KNOWN_EXTERNAL_TUPLE_STRUCTS`.

pub struct Res<T> {
    value: T,
}

impl<T> Res<T> {
    pub fn new(value: T) -> Self {
        Res { value }
    }
}

impl<T> std::ops::Deref for Res<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.value
    }
}

pub struct ResMut<T> {
    value: T,
}

impl<T> ResMut<T> {
    pub fn new(value: T) -> Self {
        ResMut { value }
    }
}

impl<T> std::ops::Deref for ResMut<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.value
    }
}

impl<T> std::ops::DerefMut for ResMut<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.value
    }
}
