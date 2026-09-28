# Parité du transpileur et des interpréteurs

Audit du 29 septembre 2026, après le premier lot de corrections numériques.

## Résultat et périmètre

Le transpileur sert de référence fonctionnelle. Les deux interpréteurs sont déjà
substantiels, mais leurs suites historiques ne couvrent pas les mêmes programmes.
Le statut « Complete » du document `interpreter-port-plan.md` décrit le portage
historique, pas une parité avec toutes les fonctionnalités actuelles du transpileur.

La comparaison automatique sélectionne les **167 fixtures simples** enregistrées
par `transpile_test!` dans `tests/transpile.rs` et disposant de fichiers `.br` et
`.expected`. Chaque programme est lancé avec son chemin réel, et stdout est comparé
à la même sortie attendue que dans la suite du transpileur.

| Résultat | Interpréteur Rust | Interpréteur Boring strict/multi |
|---|---:|---:|
| Sortie conforme | 157 | 112 |
| Erreur ou arrêt anormal | 9 | 38 |
| Sortie différente | 1 | 4 |
| Délai de 4 secondes dépassé | 0 | 13 |

**Ces nombres mesurent ce corpus, pas un pourcentage de fonctionnalités du langage.**
Un dépassement de délai est un cas à diagnostiquer, pas une preuve de boucle infinie.
Le corpus complet n'a pas été recompilé pendant cet audit : les sorties `.expected`
constituent son oracle. Le Rust généré a été réellement compilé/exécuté pour les
cas numériques modifiés (voir validation ci-dessous). Les programmes projet, les
cas négatifs, les cibles GPU et les appels à des bibliothèques Rust externes ne
sont pas couverts exhaustivement par cette sélection.

Avant correction, sur les 166 fixtures alors sélectionnées : Rust 154 conformes,
Boring 108 conformes. Le nouveau cas numérique ajoute une réussite aux deux moteurs ;
les corrections rétablissent deux anciens cas Rust et trois anciens cas Boring.
`ord_chr`, également corrigé côté Boring, appartient à la suite fonctionnelle mais
n'est pas enregistré comme fixture simple dans la suite du transpileur.

## Écarts à implémenter

| Domaine | Interpréteur Rust | Interpréteur Boring | Travail restant / reproduction |
|---|---|---|---|
| Puissances numériques | Corrigé : `.pow` entier manquait | Corrigé : `.pow` entier et `.pow`/`.powf` flottants manquaient | Cas `pow_method_*`, `numeric_method_parity` |
| `ord` / `chr` | Déjà exécutables | Corrigé : noms enregistrés sans branche dans `call_native` | Unicode BMP et hors BMP dans `numeric_method_parity` |
| `'atomic` | Lecture/affectation ordinaires possibles, mais `.swap` absent sur le scalaire | Qualificateur absent de `OwnerQual`, syntaxe refusée ; `.swap` absent | `atomic_explicit_ops`, `atomic_promotion_binding_permission_regression` ; préserver aussi partage, droits de mutation et valeur précédente |
| `'observed` | Pas de comportement de souscription sur les objets | AST/parseur sans suffixe correspondant ; runtime absent | `observed_qualifier` : callbacks, désinscription et portée des subscriptions |
| Qualificateurs, injection et valeurs partagées | `singleton_di` et retours actor/guard rejetés à la mutation ; `inject_di` et `static_di` passent | Syntaxe des attributs/suffixes manquante ; certains programmes dépassent le délai | `singleton_di`, `actor_guard_shared_return_qualifier`, `shared_return_callsite_no_double_wrap` ; audit du transport des permissions et de l'identité des objets |
| Syntaxe récente et désucrage | Front-end partagé avec le transpileur : `desugar_labeled_array`, `desugar_array_block`, `desugar_inject` | Parseur indépendant, sans équivalent complet des passes récentes | `array_block_*_transpile`, `try_prefix_in_cond_clause_noparen`, `inline_if_else_next_line_postfix`, `qualifier_group_param` |
| Méthodes de type et génériques | La plupart des cas passent ; `monomorphize_mut_arg` perd une permission de mutation | Plusieurs erreurs de parsing/binding ou dépassements du délai | `enum_type_def*`, `type_def_typed_throws`, `monomorphize_*` ; ne pas confondre effacement des types et absence de substitution/binding |
| Dictionnaires et optionnels | Cas concernés conformes | Lecture d'une clé absente jette au lieu de fournir l'optionnel attendu ; différences avec `else` | `dict_index_optional_return`, `if_let_dict_index_no_else`, `dict_string_key_index` |
| Coercitions numériques | Cas concernés conformes | Rétrécissement, débordement/cast et propagation float32 incomplets ; `atan2` absent | `narrowing_cast_if_let`, `int_literal_overflow_cast`, `if_else_cast_numeric`, `float32_struct_method_math` |
| Erreurs natives | `Error` présent | `Error` non enregistré | `builtin_error_enum`, `typed_catch_match_error` ; ajouter les variantes et la propagation typée |
| JSON et attributs serde | Implémentation dédiée dans `src/interpreter/json.rs` ; fixtures conformes | Pas de dispatch `json`/`fromJson` équivalent ; les fixtures s'arrêtent déjà sur `@` | `json_serde_shapes`, `json_serde_rename`, `json_untagged_enum` ; parser les attributs puis porter la matérialisation typée |
| Introspection | Implémentée, mais une représentation affichée diffère de l'oracle | `.introspect()` absent | `introspect_thread_safety` : Rust affiche `1.5` / `"origin"` là où l'oracle attend `Float(1.5)` / `Str("origin")` ; vérifier le contrat avant correction |
| Imports et dépendances | Sources Boring, stdlib embarquée et dépendances résolues par le front-end | Imports de fichiers présents ; `boring.*` explicitement refusé, pas de résolution `[deps]` équivalente | `boring_stdlib_collections` ; `stdlib.br::exec_use` publie aussi tous les symboles globalement, même pour un import sélectif |
| Interop Rust | Émulation limitée : `mem.swap/replace/take`, `Box.pointee`, `Duration.as_secs` manquent dans ces cas | Pas d'équivalent général à `Value::RustType` | `mem_borrow_builtins`, `pointee`, `const_promotion_known_fn` ; définir une liste explicite d'opérations émulées |
| Macros | `env!` et `include_str!` rendent une chaîne vide ; macro inconnue renvoie le dernier argument ou Void | Plusieurs macros ne sont pas dispatchées ; fallback Nil | Inspection de `src/interpreter/call.rs::call_macro` et `eval.br::eval_expr` ; définir la sémantique de runtime des macros de compilation et rejeter les autres explicitement |
| Concurrence | Futures exécutées synchroniquement, annulation non opérationnelle, délais partiellement simulés | Tâches synchrones et délai ignoré | Inspection `eval_expr.rs`, `methods.rs`, `eval.br` ; les sorties de petites fixtures de tâches ne prouvent pas une concurrence réelle |
| GPU | Simulateur CPU avec tests dédiés | Émulation propre, non comparée exhaustivement ici | Conserver la distinction entre équivalence fonctionnelle simulée et exécution GPU réelle ; utiliser les suites GPU séparées |

Les listes de manques ci-dessus reposent soit sur les fixtures en échec, soit sur
les branches explicites du code indiquées. Une ligne « inspection » n'a pas été
validée par une comparaison dynamique exhaustive.

## Ordre proposé

1. **Sécuriser le parseur Boring** : diagnostiquer les 13 dépassements de délai,
   puis aligner méthodes de type, attributs, suffixes et syntaxe des blocs.
2. **Aligner les comportements usuels** : dictionnaires optionnels, coercitions,
   fonctions mathématiques, `Error`, résolution des paramètres génériques et noms
   de méthodes (voir `builtin_name_user_members`). Ajouter chaque fixture réparée
   à `tests/run.rs` et/ou `tests/interpreter_functional.rs`.
3. **Aligner les permissions et le partage** : `mut`, retours qualifiés,
   `'atomic`, puis `'observed`. Tester les alias et les mutations visibles par
   plusieurs références, pas seulement les valeurs finales d'une variable locale.
4. **Porter les services du front-end/runtime Rust** vers Boring : désucrages,
   imports stdlib/dépendances, JSON, introspection. Garder le transpileur inchangé
   sauf si une comparaison met en évidence un défaut de la référence.
5. **Décider du contrat d'exécution** : concurrence/délais réels ou simulation
   explicitement documentée, et périmètre de l'interop Rust/GPU. Ces sujets
   demandent des tests observant l'ordonnancement, pas uniquement stdout.

## Premier lot réalisé

- `.pow` sur les douze variantes entières dans les deux interpréteurs : résultat
  de même largeur, calcul entier exact, exposant converti en `uint32` comme dans
  le transpileur. Les dépassements sont signalés par les interpréteurs via
  `checked_pow`, sans arrondi en flottant. Cela ne promet pas de reproduire le
  wrapping d'un binaire Rust compilé sans vérification de débordement.
- `.pow` et `.powf` sur float32/float64 dans l'interpréteur Boring, en conservant
  la largeur du récepteur.
- `ord` et `chr` dans son dispatch natif, avec vérification des entrées avant
  appel aux primitives du langage.
- Réutilisation des cas existants et ajout de `numeric_method_parity`, incluant
  les douze largeurs, zéro, base négative, exposant typé, entier supérieur à 2⁵³,
  méthode utilisateur homonyme, caractères accentués et emoji.
- Test unitaire Rust de conservation de variante, débordement et arguments invalides.

Validation : 845 tests unitaires, 195 tests de programmes, 4 constructions de
l'interpréteur Boring et 88 tests fonctionnels (chaque cas sur les 4 variantes).
Les 3 cas `pow_method_*` passent leurs 12 tests de transpilation/exécution.
Le nouveau cas `numeric_method_parity` passe aussi les 4 configurations du transpileur.
Le nouvel outil compare également les 4 cas numériques aux 4 interpréteurs Boring,
à l'interpréteur Rust et à un exécutable Rust nouvellement généré en strict/multi.

## Reproduire et suivre la progression

```sh
cargo build
cargo test --test interpreter_build
python3 tools/interpreter_parity.py --timeout 4 --output /tmp/parity.json
# Les écarts connus entraînent volontairement un code de sortie 1.

# Comparaison ciblée, incluant une vraie compilation de la référence :
python3 tools/interpreter_parity.py --all-modes --transpile numeric_method_parity pow_method_int_unaffected pow_method_int_exponent_var pow_method_float_width

cargo test --bin boring --test run
cargo test --test interpreter_functional
cargo test --test transpile pow_method
```

`--transpile` vérifie strict/multi ; la suite Rust `tests/transpile.rs` reste la
référence pour la matrice complète mode/threading. L'outil produit les diagnostics
et sorties divergentes dans le JSON, sans maintenir une liste d'exclusions qui
ferait artificiellement disparaître les manques. Le délai peut être augmenté
pour distinguer lenteur et blocage sur une autre machine.

## Inventaire des écarts observés après corrections

« erreur » inclut les erreurs propres et les panics de l'hôte ; consulter le JSON
pour les distinguer. « délai » signifie seulement que 4 secondes ont été dépassées.
Les 112 cas conformes sur les deux moteurs sont omis de cette table.

| Fixture dans `tests/cases/` | Rust | Boring strict/multi |
|---|---|---|
| `trait_type_level_methods` | conforme | délai |
| `array_block_flat_transpile` | conforme | erreur |
| `array_block_for_transpile` | conforme | erreur |
| `boring_stdlib_collections` | conforme | erreur |
| `int_literal_overflow_cast` | conforme | sortie différente |
| `builtin_error_enum` | conforme | erreur |
| `typed_catch_match_error` | conforme | erreur |
| `type_def_typed_throws` | conforme | délai |
| `type_method_throws_untyped` | conforme | délai |
| `untyped_string_lit_local_to_type_method` | conforme | délai |
| `enum_type_def` | conforme | erreur |
| `enum_type_def_throws` | conforme | erreur |
| `float32_struct_method_math` | conforme | erreur |
| `pointee` | erreur | erreur |
| `pub_top_level_const` | conforme | erreur |
| `const_promotion_known_fn` | erreur | erreur |
| `dict_index_optional_return` | conforme | erreur |
| `if_let_dict_index_no_else` | conforme | erreur |
| `dict_string_key_index` | conforme | sortie différente |
| `builtin_name_user_members` | conforme | sortie différente |
| `implicit_self_length_nontail` | conforme | délai |
| `throws_method_name_collision` | conforme | délai |
| `narrowing_cast_if_let` | conforme | sortie différente |
| `try_prefix_in_cond_clause_noparen` | conforme | erreur |
| `option_owned_methods` | conforme | délai |
| `cast_bare_field_index` | conforme | délai |
| `if_else_cast_numeric` | conforme | erreur |
| `mem_borrow_builtins` | erreur | erreur |
| `json_serde_rename` | conforme | erreur |
| `try_wrap_double_handling` | conforme | erreur |
| `json_untagged_enum` | conforme | erreur |
| `json_serde_shapes` | conforme | erreur |
| `enum_derive_no_debug` | conforme | erreur |
| `self_field_loop_match_borrow` | conforme | délai |
| `introspect_thread_safety` | sortie différente | erreur |
| `qualifier_group_param` | conforme | erreur |
| `enum_variant_shadow` | conforme | erreur |
| `monomorphize_struct` | conforme | délai |
| `monomorphize_mut_arg` | erreur | erreur |
| `monomorphize_cross_file_main` | conforme | délai |
| `monomorphize_method` | conforme | erreur |
| `monomorphize_method_on_generic_struct` | conforme | erreur |
| `monomorphize_optional_method` | conforme | erreur |
| `monomorphize_ext_method` | conforme | erreur |
| `monomorphize_enum_method` | conforme | erreur |
| `guard_let_else_panic_throws` | conforme | erreur |
| `atomic_explicit_ops` | erreur | erreur |
| `atomic_promotion_binding_permission_regression` | erreur | erreur |
| `observed_qualifier` | erreur | erreur |
| `singleton_di` | erreur | erreur |
| `inject_di` | conforme | délai |
| `static_di` | conforme | délai |
| `actor_guard_shared_return_qualifier` | erreur | erreur |
| `shared_return_callsite_no_double_wrap` | conforme | erreur |
| `inline_if_else_next_line_postfix` | conforme | erreur |
