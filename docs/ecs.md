# Native ECS Integration in Boring

---

## 1. Vision and Goals

Boring integrates natively with Bevy (`breakout-boring`, `boring-bevylib`). This native integration turns **ECS into a first-class language citizen** across Boring's syntax and semantics:

1. **Zero Syntactic Noise**: No manual Bevy generic wrappers (`Query`, `Res`, `ResMut`, `Single`), system parameter wiring, or manual query iterators in user code.
2. **Pure Value Types (`data`)**: A clear distinction between general objects (`struct`) and pure value data (`data`), strictly forbidding shared pointer qualifiers (`'actor`, `'shared`) to guarantee CPU cache locality, zero pointer indirection, and thread safety.
3. **Entity Components (`@component data`)**: Clean, explicit component declarations.
4. **Simple Resources (`@resource`) & Direct Signatures**: Global resources accessed and mutated naturally (`score += 10`) and passed cleanly into system signatures (`Score score` or `mut Score score`).
5. **Universal Systems (`def` / `req` with `where`)**:
   * **Component Method Systems**: `self` represents the primary component being evolved (`def movement(Velocity vel) where not Frozen:`). Receiver mutability follows directly from `def` (mutable) or `req` (read-only) without any `mut self` syntax.
   * **Top-Level Systems**: Mediator systems operating across distinct components (`def check_collisions(Score score):`).
6. **Inline Query Expressions (`query`)**:
   * `query (Entity, Transform) where Brick` (or `query all ...`) for zero-allocation iterable views.
   * `query single (mut Transform) where Paddle` for a single required entity.
   * `query single? (Transform) where Bonus` for an optional single entity.
   * `query first / first?` for the first matching entity.
   * `query any where Ball` for an immediate boolean existence test.
7. **Ambient Context & Infrastructure**: Direct natural access to frame timing (`time.delta`), inputs (`input.pressed(...)`), commands (`commands.spawn(...)`), current entity ID (`entity`), and current entity destruction (`despawn()`), with qualified fallbacks (`ecs.entity`, `ecs.despawn()`, `ecs.time`, `ecs.input`, `resource.<name>`) when shadowed by local variables.
8. **Unified Modular Imports (`use ecs.<module>`)**: Framework-agnostic import syntax (`use ecs.asset.RenderAssetUsages`, `use ecs.input.keyboard.KeyboardInput`, etc.).

---

## 2. Declaring Data: `data` and `@component`

### 2.1 The `data` Keyword (Pure Value / POD)

The `data` keyword defines a pure value data type:
* **Allowed members**: Scalars (`int`, `float`, `bool`), strings (`string`), fixed-size arrays/vectors (`[float, 2]`, `[float, 3]`, `[int]`), spatial types (`Transform`), Boring enums, optionals (`Target?`), and nested `data` types.
* **Forbidden members**: Shared pointer qualifiers (`'actor`, `'shared`, internal locks, local references). The compiler statically guarantees that every `data` type is strictly `Send + Sync + 'static` and contiguous in memory.

### 2.2 Entity Components (`@component data`)

Decorating a `data` declaration with `@component` marks it as a component that can be attached to ECS entities:

```boring
@component
data Position:
    float x = 0.0
    float y = 0.0

@component
data Velocity:
    float dx = 0.0
    float dy = 0.0

# Zero-sized unit tag component (0 bytes, used for archetype filtering)
@component
data Frozen: pass
```

**Rust / Bevy Lowering:**
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

## 3. Global Resources: `@resource`

Global application state (game score, configuration, persistent state) is declared using `@resource` on global variables or `data` types:

```boring
@resource
var int score = 0

@resource
float gravity = -9.81

@resource
data GameConfig:
    var int max_players = 4
```

### Usage in Systems
In any system, a resource can be read or mutated directly by name (e.g. `score += 10`), qualified via `resource.<name>` or `ecs.resource.<name>` when shadowed by a local variable, or passed explicitly as a system parameter (`Score score` or `mut Score score`):

```boring
def kill_enemy(Enemy e) where e.is_dead:
    score += 10
    despawn()

def sync_layers(mut Transform transform, ScratchSprite tag):
    let order = layerOrders.get(tag.target_id)
    transform.translation.z = order as float32

def update_scoreboard(Score score, mut TextUiWriter writer):
    let score_root = query single entity where (ScoreboardUi, Text)
    writer.text(score_root, 1).pointee = "{score.count}".to_string()
```

**Rust / Bevy Lowering:**
The compiler statically checks usage (read vs. mutation):
* For `@resource var int score = 0`, it emits `#[derive(Resource)] pub struct Res_score(pub isize);`.
* Read-only usage injects `score: Res<Score>` or `res_layer_orders: Res<LayerOrders>`.
* Reassignment or mutation injects `mut score: ResMut<Score>`.
* Infrastructure types (`Commands`, `TextUiWriter`, etc.) are injected directly without redundant generic wrappers.

---

## 4. Declaring Systems: `def` and `req` with `where`

Systems are declared with standard function keywords: `def` (mutating systems) or `req` (read-only systems), along with an optional `where` filter clause.

### 4.1 Component Method Systems

When a system primarily modifies or queries a single component, declare it directly inside that component's `data` block:

```boring
@component
data Position:
    float x
    float y

    @schedule(Update)
    def movement(Velocity vel) where not Frozen:
        self.x += vel.dx * time.delta
        self.y += vel.dy * time.delta
```

* As with any Boring method, receiver mutability is dictated by the keyword: `def` automatically grants mutable access (`&mut Position` in Bevy), while `req` grants read-only access (`&Position`). Explicit `mut self` is neither needed nor valid syntax.
* `(Velocity vel)` lists secondary components required for the calculation.
* `self` can be omitted from the parameter list or written simply as `self`.
* For read-only systems on the component, use `req movement(Velocity vel): ...`.

**Rust / Bevy Lowering:**
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

### 4.2 Top-Level Systems

For mediator systems operating across multiple entity types or handling system lifecycle events:

```boring
# Physics mediator system
def apply_velocity(mut Position pos, Velocity vel) where not Frozen:
    pos.x += vel.dx * time.delta
    pos.y += vel.dy * time.delta

# Startup system (runs once on launch, commands is injected automatically)
@schedule(Startup)
def setup_world():
    commands.spawn((Position(x = 0.0, y = 0.0), Velocity(dx = 10.0, dy = 5.0)))

# Entity lifecycle system
def death_check(Health h) where h.current <= 0:
    despawn()
```

---

## 5. Inline Query Expressions: `query`

Inside systems, the `query` keyword performs entity lookups and provides zero-allocation iterable views:

### 5.1 Entity Sets (`query` / `query all`)
Returns an iterable Bevy query view without temporary heap allocations (`Vec`):
```boring
# Concise syntax (equivalent to 'query all'):
let bricks = query (Entity, Transform) where Brick

# Explicit syntax:
let all_bricks = query all (Entity, Transform) where Brick

# O(1) empty check:
if bricks.isEmpty():
    return

# Iteration:
for b in bricks:
    ...
```

### 5.2 Single Entity (`query single` and `query single?`)
Requires exactly one matching entity (panics or errors if 0 or >1), or returns `None` with `?`:
```boring
mut ball = query single (mut Velocity, Transform) where Ball
let maybe_bonus = query single? (Transform) where Bonus
```

### 5.3 First Matching Entity (`query first` and `query first?`)
Retrieves the first matching entity found:
```boring
let leader = query first (Entity, Transform) where Player
```

### 5.4 Existence Check (`query any`)
Evaluates directly to a boolean indicating whether at least one matching entity exists:
```boring
if query any where Ball:
    print("Ball is in play")
```

---

## 6. `for ... where` Loops

Boring allows iterating directly over a set of components with filtering without intermediate variable declarations:

```boring
for entity, Transform collider_trans, Brick? maybe_brick where Collider:
    if maybe_brick != nil:
        commands.entity(entity).despawn()
```

---

## 7. Scheduling and Lifecycle: `@schedule` and `@state`

1. **`@schedule(...)`**: Configures the execution stage per frame.
   * Standard schedules: `Startup`, `FixedUpdate`, `Update` (default if omitted), `PostUpdate`, `Render`.
   * Example: `@schedule(FixedUpdate)` for constant-timestep physics (50/60 Hz).
2. **`@state(...)` (or `@inState(...)`)**: Restricts system execution to a specific application state.
   * Example: `@state(GameState.Playing)` (lowered to `.run_if(in_state(GameState::Playing))`).

---

## 8. Archetype and Value Filtering: `where`

The `where` clause defines fine-grained constraints on archetypes and values:
* **Component presence**: `where Player` $\rightarrow$ `With<Player>`
* **Component absence**: `where not Frozen` $\rightarrow$ `Without<Frozen>`
* **Boolean combinations**: `where Player and not (Frozen or Stunned)`
* **Change detection**: `where changed Transform` $\rightarrow$ `Changed<Transform>`
* **Addition detection**: `where added Enemy` $\rightarrow$ `Added<Enemy>`
* **Value filtering**: `where h.current <= 0` $\rightarrow$ condition evaluated at the start of each entity iteration step.

---

## 9. Ambient Context and Conflict Disambiguation

Inside ECS systems, Boring automatically detects and injects infrastructure services (`time`, `input`, `commands`, and `@resource` variables) without cluttering the function signature.

### 9.1 Direct Default Access (Recommended)

By default, ambient services and resources are accessed directly by their natural name:

| Boring Expression | ECS Meaning | Generated Rust / Bevy Equivalent |
|---|---|---|
| `time.delta` / `time.deltaSecs()` | Elapsed frame delta time (seconds) | `(time.delta_secs() as f64)` (via `Res<Time>`) |
| `time.elapsed` / `time.elapsedSecs()` | Total elapsed simulation time | `time.elapsed_secs_f64()` |
| `input.pressed(...)` | Key / button held state | `input.pressed(...)` (via `Res<ButtonInput<KeyCode>>`) |
| `input.justPressed(...)` | Key pressed this frame | `input.just_pressed(...)` |
| `input.justReleased(...)` | Key released this frame | `input.just_released(...)` |
| `commands.spawn(...)` | Spawns an entity with components | `commands.spawn(...)` |
| `commands.entity(e).despawn()` | Despawns a specific entity | `commands.entity(e).despawn()` |
| `<resource_name>` (e.g. `score += 10`) | Reads or mutates an `@resource` variable | `res_score.0 += 10` (via `ResMut<Res_score>`) |
| `entity` | Current entity ID in query loop | `Entity` from query loop |
| `despawn()` | Despawns current entity in query loop | `commands.entity(__entity).despawn()` |

> [!NOTE]
> **Naming conventions**: In idiomatic Boring, methods and fields follow `camelCase` (`justPressed(...)`, `justReleased(...)`, `deltaSecs()`). The Boring transpiler automatically maps `camelCase` method and property names to the idiomatic `snake_case` required by Rust and Bevy (`just_pressed`, `delta_secs`). Both `camelCase` and `snake_case` are fully supported.

### 9.2 Explicit Qualification (`ecs.` and `resource.`) on Conflict

If a local variable or parameter shadows an ambient service or resource (such as a local variable named `score`, `time`, or `input`), the local variable takes precedence. You can disambiguate using qualified prefixes:

* **Current Entity**: `ecs.entity`
* **Current Entity Despawn**: `ecs.despawn()`
* **Resources**: `resource.score` or `ecs.resource.score` (or `resource.layerOrders`)
* **Time**: `ecs.time.delta` or `ecs.time.elapsed`
* **Input**: `ecs.input.pressed(KeyCode.Space)`
* **Commands**: `ecs.commands.spawn(...)`

Legacy leading-dot syntax (`.entity`, `.despawn()`, `.time`, `.input`, `.commands`, `.resource.<name>`) remains supported inside ECS systems for backward compatibility.

---

## 10. Modular Imports: `use ecs.*`

Specialized ECS types and subsystems (rendering, audio, input, sprites) are imported using the standard `ecs.*` namespace:

```boring
use ecs.asset.RenderAssetUsages
use ecs.render.Extent3d, TextureDimension, TextureFormat
use ecs.audio.Volume
use ecs.input.keyboard.KeyboardInput
use ecs.input.ButtonState
use ecs.sprite.Anchor
```

---

## 11. The `@game` Decorator and Game Framework

To write 100% native Boring games without any Bevy boilerplate (`App.new()`, `use bevy`, `DefaultPlugins`, manual main loop), Boring provides the `@game` decorator on a `data` or `struct` definition:

```boring
@game
pub data Breakout:
    string title = "My Awesome Game"
    int width = 800
    int height = 600
    Color clear_color = Color.srgb(0.9, 0.9, 0.9)
```

### 11.1 Automatic Application Bootstrapping

When `@game` is present and no explicit `def main()` exists:
1. **Window Configuration**: `WindowPlugin` is configured with `width`, `height`, window `title`, and `clear_color`.
2. **Automatic 2D Camera**: If no `@schedule(Startup)` system spawns a camera (`Camera2D` or `Camera3D`), a default 2D camera (`Camera2d::default()`) is injected.
3. **Resource Registration**: All `@resource` declarations and the `@game` struct itself are initialized (`app.init_resource::<T>()`).
4. **System Scheduling**:
   * Systems decorated with `@schedule(Startup)` are added to `Startup`.
   * Systems decorated with `@schedule(Update)` (or without schedule attributes in a `@game` project) are added to `Update`.
   * Systems decorated with `@schedule(FixedUpdate)` are added to `FixedUpdate`.
   * Ordering constraints are set via `@after(other_system)` or `@before(other_system)`.
   * Event observers (`On<Event>` / `Trigger<Event>` or `@observer`) are registered via `app.add_observer(...)`.
5. **Generation of `build_app()` and `main()`**: The compiler synthesizes both `pub fn build_app() -> App` (for headless testing and external integration) and `fn main() { build_app().run(); }`.

### 11.2 Built-in Standard Components

* **`Camera2D` / `Camera2d`**: Standard 2D camera component.
* **`Sprite`**: Standard 2D sprite rendering component.
* **`Transform`**: Direct coordinate access (`.x`, `.y`, `.z` both for reading and reassignment `.x = ...`) automatically translated to `.translation.x`, `.translation.y`, `.translation.z` while preserving Bevy's `f32` typing.
