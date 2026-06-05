# Draft — Mapping Boring → Rust-for-Linux

> État : analyse préliminaire — non implémenté

---

## Primitives

| Boring | Rust std | Rust-kernel | Statut | Notes |
|--------|----------|-------------|--------|-------|
| `int` | `i64` | `i64` | ✅ | identique |
| `uint` | `u64` | `u64` | ✅ | identique |
| `float` | `f64` | — | ❌ | FPU interdite dans le kernel (sauf cas explicites) |
| `bool` | `bool` | `bool` | ✅ | identique |
| `string` | `Arc<str>` | `kernel::str::CStr` / `CString` | ⚠️ | strings kernel C-compatible |
| `void` | `()` | `()` | ✅ | identique |

---

## Types composés

| Boring | Rust std | Rust-kernel | Statut | Notes |
|--------|----------|-------------|--------|-------|
| `T?` | `Option<T>` | `Option<T>` | ✅ | disponible dans `core::` |
| `[T]` | `Vec<T>` | `kernel::prelude::Vec<T>` | ✅ | kernel a son propre Vec avec allocateur kernel |
| `{K: V}` | `HashMap<K,V>` | — | ❌ | pas de HashMap, utiliser `kernel::rbtree::RBTree` |
| `{T}` | `HashSet<T>` | — | ❌ | pas d'équivalent direct |
| `(T, U)` | tuples | tuples `core::` | ✅ | |
| `Box<T>` | `Box<T>` | `Box<T, KernelAllocator>` | ⚠️ | allocateur différent |

---

## Ownership qualifiers

| Boring | Rust std | Rust-kernel | Statut | Notes |
|--------|----------|-------------|--------|-------|
| `T'` | `Box<T>` | `Box<T>` | ✅ | avec allocateur kernel |
| `T'auto` | `Rc<T>` | `kernel::sync::Arc` | ⚠️ | remplacé par Arc<T> (Rc non dispo dans le kernel) |
| `T'task` | `Arc<T>` | `kernel::sync::Arc` | ✅ | |
| `T'actor` | `Arc<Mutex<T>>` | `kernel::sync::Mutex` | ✅ | |
| `T'guard` | `Arc<RwLock<T>>` | `kernel::sync::RwLock` | ✅ | |
| `T'weak` | `Weak<T>` | via `kernel::sync::Arc` | ✅ | |
| `T'stack` | `T` | `T` | ✅ | |
| `T&` / `var T&` | `&T` / `&mut T` | `&T` / `&mut T` | ✅ | |

---

## Fonctions et error handling

| Boring | Rust std | Rust-kernel | Statut | Notes |
|--------|----------|-------------|--------|-------|
| `throws` | `Result<T, Box<dyn Error>>` | `Result<T, kernel::error::Error>` | ⚠️ | type d'erreur fixe (errno-based) |
| `throws MyError` | `Result<T, MyError>` | `Result<T, kernel::error::Error>` | ⚠️ | erreurs custom à mapper sur des errnos |
| `try/catch` | pattern matching | pattern matching | ✅ | |
| `guard … else throw` | early return | early return | ✅ | |

---

## Async / concurrence

| Boring | Rust std | Rust-kernel | Statut | Notes |
|--------|----------|-------------|--------|-------|
| `task fn` | `async fn` + tokio | `workqueue` / `kthread` | ❌ | pas d'async/await tokio |
| `task expr` | `tokio::spawn` | `kernel::workqueue::Work` | ⚠️ | mapping possible, sémantique différente |
| `stream fn` | `futures::Stream` | — | ❌ | pas d'équivalent |
| `chan` / `tx.send` / `rx.recv` | MPSC tokio | — | ❌ | utiliser `kernel::sync::CondVar` ou spinlock |
| `select:` | `tokio::select!` | — | ❌ | pas d'équivalent |
| `wait Duration` | `tokio::sleep` | `kernel::delay::coarse_sleep` | ⚠️ | disponible mais sans await |
| `Future<T>` | `tokio::JoinHandle` | — | ❌ | |

---

## Stdlib

| Boring | Rust std | Rust-kernel | Statut | Notes |
|--------|----------|-------------|--------|-------|
| `print!` | `println!` | `pr_info!` / `pr_err!` | ⚠️ | macros kernel différentes |
| `assert_eq!` | `assert_eq!` | `kernel::build_assert!` | ⚠️ | panics interdits → `WARN_ON` |
| `panic(msg)` | `panic!` | — | ❌ | un panic = kernel oops/crash |
| Math (`sqrt`, `sin`…) | `std::f64` | — | ❌ | FPU interdite |
| `Vec` methods | `std::vec` | `kernel::prelude::Vec` | ⚠️ | API légèrement différente |
| `HashMap` | `std::collections` | — → `RBTree` | ❌ | remplacement nécessaire |

---

## Synthèse

### Ce qui mappe bien (~60% du langage)

- Tous les primitifs sauf `float`
- Ownership qualifiers (`Box`, `Arc`, `Mutex`, `RwLock`)
- Structs, enums, traits, génériques
- Pattern matching, control flow
- Error handling (avec adaptation des types d'erreur)
- `Vec`, tuples, `Option`

### Ce qui nécessite un mapping alternatif

- `string` → `CStr` / `CString` kernel
- `HashMap` / `HashSet` → `RBTree` ou absent
- `throws MyError` → `kernel::error::Error` (errno-based)
- `print!` / assertions → macros kernel
- `Box<T>` → allocateur kernel

### Ce qui est incompatible / à désactiver

- Tout l'async : `task`, `stream`, `select`, `Future`, channels (pas de tokio — remplacer par workqueue/kthread si besoin)
- `float` et toute la math flottante
- `panic`
- `T'auto` (remplacé par `Arc<T>` — non bloquant mais moins optimal qu'en Boring standard)

---

## Architecture envisagée pour le transpileur kernel

- **Partagé** : parser, AST, passes de typage
- **Nouveau** : passe de validation (liste noire des constructs incompatibles) + émetteur `emit_kernel_*.rs`
- L'émetteur duplique et adapte les fichiers `emit_top.rs`, `emit_stmt.rs`, `emit_expr.rs`, `helpers.rs`
- Le flag `--emit-rust-kernel` sélectionne l'émetteur au moment de la compilation

---

## Prochaines étapes (à décider)

1. Passe de validation — rejeter float, async, panic, Rc, HashMap avec messages d'erreur explicites
2. Nouvel émetteur — substitution des types et macros kernel
3. Stdlib kernel — sous-ensemble de la stdlib Boring compatible `no_std`
