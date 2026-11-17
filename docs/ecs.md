# Intégration ECS Native dans le Langage Boring

> **Statut** : Implémenté et Validé  
> **Auteur** : Mickaël LANOË & Assistant  
> **Cible initiale (Phase 1)** : Abaissement direct vers Bevy ECS en Rust  

---

## 1. Vision et Objectifs

Boring dispose d'intégrations avec Bevy (`bevy-boring`, `breakout-boring`, `boring-bevylib`). L'intégration native fait de l'**ECS un citoyen de première classe** dans la syntaxe et la sémantique de Boring :
1. **Élimination du bruit syntaxique** : Pas de types génériques Bevy ni d'itérateurs manuels dans le code utilisateur.
2. **Types de données pures (`data`)** : Distinction claire entre objets généraux (`struct`) et données par valeur (`data`), interdisant formellement les pointeurs partagés (`'actor`, `'shared`) pour préserver la localité du cache et le parallélisme.
3. **Composants (`@component data`)** : Déclaration explicite et lisible des composants d'entités.
4. **Ressources simples (`@resource`)** : Variables globales sans bloc struct ni double nom, utilisables comme n'importe quelle variable (`score += 10`).
5. **Queries universelles (Méthodes et Fonctions)** :
   * **Méthodes dans un composant** : `self` représente le composant principal à faire évoluer (`query movement(Velocity vel): self.x += ...`).
   * **Fonctions top-level** : Pour les systèmes médiateurs opérant sur plusieurs composants distincts (`query check_collisions(Transform a, ...):`).
6. **Contexte ambiant avec point (`.`)** : Accès direct au temps (`.time.delta`), aux entrées (`.input`), à l'entité courante (`.entity`), et aux commandes (`.despawn()`, `.spawn(...)`).

---

## 2. Déclaration des Données : `data` et `@component`

### 2.1 Le mot-clé `data` (Pure Value / POD)

Le mot-clé `data` définit un type de données pures par valeur :
* **Membres autorisés** : Scalaires (`int`, `float`, `bool`), chaînes (`string`), vecteurs mathématiques (`Vec2`, `Vec3`, `Transform`), enums Boring, tableaux (`[int]`, `[float, 3]`), optionnels (`Target?`), et autres types `data`.
* **Membres interdits** : Qualificateurs de pointeurs partagés (`'actor`, `'shared`, verrous internes, références locales). Le compilateur garantit ainsi statiquement que tout `data` est strictement `Send + Sync + 'static` et optimal pour le cache CPU.

### 2.2 Composants d'entités (`@component data`)

Pour marquer un `data` comme composant pouvant être attaché à une entité ECS :

```boring
@component
data Position:
    float x = 0.0
    float y = 0.0

@component
data Velocity:
    float dx = 0.0
    float dy = 0.0

# Tag component / Unit component (0 octet, utilisé comme filtre d'archétype)
@component
data Frozen: pass
```

**Transpilation Rust / Bevy :**
```rust
#[derive(bevy::prelude::Component, Debug, Clone, PartialEq)]
pub struct Position {
    pub x: f64,
    pub y: f64,
}

#[derive(bevy::prelude::Component, Debug, Clone, PartialEq)]
pub struct Velocity {
    pub dx: f64,
    pub dy: f64,
}

#[derive(bevy::prelude::Component, Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Frozen;
```

---

## 3. Ressources Globales : `@resource`

Les états globaux (score du jeu, configuration, réglages) sont déclarés avec `@resource` sans forcer de bloc struct inutile :

```boring
@resource
var int score = 0

@resource
float gravity = -9.81

@resource
data GameConfig:
    var int max_players = 4
```

### Usage dans les Queries
Dans n'importe quelle query, la ressource est lue ou modifiée comme une variable ordinaire :

```boring
query kill_enemy(Enemy e) where e.is_dead:
    score += 10
    .despawn()
```

**Transpilation Rust / Bevy :**
Le compilateur génère le wrapper `#[derive(Resource)] pub struct Res_score(pub isize);`, injecte `mut res_score: ResMut<Res_score>` dans la fonction système générée, et applique `res_score.0 = ...`.

---

## 4. Systèmes et Queries : `query`

Une `query` définit un système itérant sur toutes les entités satisfaisant une signature de composants et une condition de filtrage `where`.

### 4.1 Queries sous forme de méthode de composant

Lorsqu'un système fait évoluer un composant particulier, il est déclaré directement dans le `data` de ce composant :

```boring
@component
data Position:
    float x
    float y

    @schedule(Update)
    query movement(Velocity vel) where not Frozen:
        self.x += vel.dx * .time.delta
        self.y += vel.dy * .time.delta
```

* `self` est automatiquement lié au composant de l'entité courante en accès mutable (`&mut Position`).
* `(Velocity vel)` ne liste que les composants secondaires requis pour le calcul.
* Si le système ne fait que de la lecture seule sur `self`, on utilise `req query` (`&Position`).

**Transpilation Rust / Bevy :**
```rust
pub fn position_movement(
    mut query: Query<(&mut Position, &Velocity), Without<Frozen>>,
    time: Res<Time>,
) {
    for (mut self_, vel) in &mut query {
        self_.x += (vel.dx * (time.delta_secs() as f64));
        self_.y += (vel.dy * (time.delta_secs() as f64));
    }
}
```

### 4.2 Queries au niveau global (Top-level)

Pour les systèmes médiateurs opérant entre plusieurs entités ou sans composant propriétaire :

```boring
# Système de physique générale
query apply_velocity(mut Position pos, Velocity vel) where not Frozen:
    pos.x += vel.dx * .time.delta
    pos.y += vel.dy * .time.delta

# Système de cycle de vie (exécuté 1 fois au lancement, sans composant)
@schedule(Startup)
query setup_world():
    .spawn(Position(x = 0.0, y = 0.0), Velocity(dx = 10.0, dy = 5.0))

# Système de destruction
query death_check(Health h) where h.current <= 0:
    .despawn()
```

---

## 5. Ordonnancement et Cycle de Vie : `@schedule` et `@state`

1. **`@schedule(...)`** : Règle la phase d'exécution par frame.
   * Valeurs standard : `Startup`, `FixedUpdate`, `Update` (défaut si omis), `PostUpdate`, `Render`.
   * Exemple : `@schedule(FixedUpdate)` pour une physique à pas constant (50/60 Hz).
2. **`@state(...)` (ou `@inState(...)`)** : Conditionne l'exécution selon l'état applicatif.
   * Exemple : `@state(GameState.Playing)` (transpilé en `.run_if(in_state(GameState::Playing))`).

---

## 6. La Clause de Filtrage `where`

La clause `where` permet d'exprimer les contraintes fines sur les archétypes et les valeurs :
* **Présence de tag** : `where Player` $\rightarrow$ `With<Player>`
* **Exclusion de tag** : `where not Frozen` $\rightarrow$ `Without<Frozen>`
* **Combinaisons booléennes** : `where Player and not (Frozen or Stunned)`
* **Détection de changement** : `where changed Transform` $\rightarrow$ `Changed<Transform>`
* **Détection d'ajout** : `where added Enemy` $\rightarrow$ `Added<Enemy>`
* **Filtre de valeur** : `where h.current <= 0` $\rightarrow$ condition évaluée au début de chaque itération.

---

## 7. Contexte Ambiant (`.`)

À l'intérieur du corps d'une `query`, l'opérateur point (`.`) donne accès au contexte sans polluer la liste des paramètres :

| Expression Boring | Signification ECS | Équivalent Rust / Bevy généré |
|---|---|---|
| `.time.delta` | Temps écoulé depuis la dernière frame (secondes) | `(time.delta_secs() as f64)` (via `Res<Time>`) |
| `.time.elapsed` | Temps total écoulé de la simulation | `time.elapsed_secs_f64()` |
| `.input.is_key_pressed(...)` | Touche pressée | `input.pressed(...)` (via `Res<ButtonInput<KeyCode>>`) |
| `.entity` | ID de l'entité courante | Tuple avec `Entity` |
| `.despawn()` | Détruit l'entité courante | `commands.entity(__entity).despawn()` |
| `.spawn(c1, c2, ...)` | Crée une nouvelle entité avec ses composants | `commands.spawn((c1, c2, ...))` |
| `score += 10` | Modifie une ressource globale `@resource` | `res_score.0 += 10` (via `ResMut<Res_score>`) |

Le compilateur analyse statiquement le corps de la query pour détecter ces occurrences et génère automatiquement les paramètres Bevy requis (`time: Res<Time>`, `mut commands: Commands`, etc.).
