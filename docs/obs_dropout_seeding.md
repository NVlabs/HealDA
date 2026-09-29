# How observation dropout gets its randomness

Observation dropout takes **no seed argument**. It draws from the *ambient stream*, and
that one fact explains the whole design. This page exists so nobody has to re-derive it.

## "Ambient stream"

The **process-global NumPy generator** — `np.random.random()`, `np.random.randint()`, the
functions that take no generator object. One per process, so one per DataLoader worker.

Contrast with a **child generator**: `np.random.default_rng(seed)`, a private object. Every
draw is either ambient (shared, ordered, the thing that must stay aligned) or from a child
(private, unordered, free).

## Who seeds the ambient stream

```text
LoopConfig.seed  (one int for the run)
        |
        +--> [_setup_networks]  torch.manual_seed(seed)
        |         rank-INDEPENDENT, so every rank builds identical weights
        |
        v
[main process, ONCE at the top of TrainingLoopBase._train]
   np.random.seed((seed * rng_world + rng_rank + cur_nimg) % (1 << 31))
        |                              ^^^^^^^^  data rank, not global rank,
        |                                        when fix_time_parallel_rng
        |                                        cur_nimg is the value at loop
        |                                        entry: 0 new, resume point on
        |                                        resume. It does not advance.
        v
   torch.manual_seed(np.random.randint(1 << 31))
        |
        v
[DataLoader iterator]  _base_seed = torch global generator draw
        |
        |   persistent_workers=True: drawn ONCE, at first spawn.
        |   Later epochs call _reset(), which does not reseed workers.
        |
        +---------------------+---------------------+
        v                     v                     v
   worker 0              worker 1              worker 2
   np.random.seed(_generate_state(base_seed, worker_id))
        |                     |                     |
        v                     v                     v
   THE AMBIENT STREAM (one process-global numpy generator per worker)
        |
        |   Each loader spends exactly ONE draw off it per sel_time:
        |       parent = default_rng(np.random.randint(2**31))
        v
   satellite -> conventional -> satwnd      (static gather order)
        |
        v
   everything else comes off `parent`, never off the ambient stream:
     - the sample-scope verdict          parent.random()
     - one seed per archive file         parent.integers()  -> thread_pool()
```

Across the mesh, for one sample:

```text
             data rank 0                        data rank 1
       +-----------------------+          +-----------------------+
       | tp0  tp1  tp2  tp3    |          | tp0  tp1  tp2  tp3    |
       | frames 0-1, 2-3, ...  |          | a different sample    |
       +-----------------------+          +-----------------------+
         SAME ambient seed                  DIFFERENT ambient seed
         (rng_rank = data rank)             (data rank differs)
                |                                    |
         same draws, same count              independent draws
                |                                    |
         same sample verdict                  its own verdict
         with no communication
```

Four things the diagram cannot show, all of which matter:

- **The seed is used twice, for opposite purposes.** `_setup_networks` calls
  `torch.manual_seed(self.seed)` with no rank term, so every rank initialises the same
  weights. `_train` then re-seeds with a rank term, so the ranks diverge for everything
  stochastic that follows. Reading either one alone gives the wrong picture: the same
  `seed` field means "be identical" at construction and "be different" at training.
- **`worker_id` is local.** It is the worker's index within its own DataLoader,
  `0..num_workers-1`, not a global id. That is why worker 1 on tp rank 0 and worker 1 on tp
  rank 2 start identical rather than merely independent.
- **The seed is set once per process, not per step.** `_train()` calls
  `np.random.seed(...)` before entering its loops, so `cur_nimg` there is the value at
  loop entry. It distinguishes a fresh run from a resume, never one step from the next.
- **`persistent_workers=True` means seeded once, ever.** The iterator is created once in the
  DataLoader's lifetime; later epochs call `_reset()`, which resets task bookkeeping but does
  not respawn workers or resend a seed. A worker's stream is seeded at first spawn and only
  advances after that. So nothing recomputed in the main process — including `cur_nimg` —
  reaches a worker again.

## The alignment invariant

> Every loader draws **exactly once** from the ambient stream per `sel_time`, and nothing
> else in the worker touches it.

Draw count is a function of **config** (which loaders exist), never of **data**. So the
identical streams above stay identical, and ranks that never communicate reach the same
answer. That is the entire mechanism behind sample-scoped dropout: no key is threaded and
nothing is broadcast — four ranks independently compute the same number.

Each loader spends its one draw on a child, and everything else comes off that child:

```python
parent = np.random.default_rng(np.random.randint(2**31))   # the only ambient draw
rates  = {f: float(parent.random() < p) for f in families} # sample verdict
seeds  = {unit: int(parent.integers(2**31)) for unit in needed}
```

The per-unit seeds exist for **thread safety**, not statistics: each archive file is
decoded on its own `thread_pool()` thread, and a shared `Generator` is not thread-safe. The
unit is whatever the archive is keyed by — a 6-hourly cycle file for PrepBUFR and GPS-RO, a
day file for SATWND and the satellite sensors.

### Why the draws cannot be reordered

- **All ambient draws are on the event-loop thread.** In every `sel_time` there is no
  `await` between `async def` and the draw. asyncio is cooperative, so a coroutine runs
  uninterrupted to its first `await`; nothing can interleave before a draw. The pool threads
  only ever touch the child they were handed.
- **Loader order is static**, fixed by `asyncio.gather` argument order in
  `CombinedObsLoader.sel_time`: satellite, then conventional, then SATWND (whose coroutine
  body runs only when awaited). Identical on every rank.
- **Worker assignment matches across ranks.** The sampler is sharded on the data rank, so
  all time-parallel ranks iterate the same sample sequence, and round-robin hands sample `i`
  to worker `i % num_workers` everywhere.

### What breaks it

Adding a data-dependent ambient draw anywhere in the worker — a conditional
`np.random.*` in a loader, a transform, or a library. Then ranks whose slices differ consume
different counts, their streams separate permanently, and sample scope silently degrades to
per-rank-slice dropout with no error.

This was a real bug, not a hypothetical: seeds were once drawn per cycle *and* per day
directly off the ambient stream, so a rank whose frames straddled midnight took two extra
draws. Half of all sample alignments diverged.

`test_every_time_parallel_rank_consumes_the_same_ambient_draws` pins the invariant across
all four alignments. It is the only thing standing between this design and a silent failure,
so it should not be deleted or marked xfail.

## Scopes

`ObsConfig.nnja_dropout_scope` governs **every** NNJA random drop — wind and satellite
platform/channel alike.

| scope | behaviour | needs alignment? |
|---|---|---|
| `"row"` (default) | each observation independently | no |
| `"sample"` | a whole family/channel present or absent for the entire sample | yes |

`"sample"` collapses a rate to 0.0 or 1.0 with one draw and reuses the existing
all-or-nothing branches, so it adds no new masking path.

Families draw separately, so AMVs, aircraft winds and scatterometers decide independently,
as do two channel rules on one platform. Note AMVs come only from the SATWND archive —
PrepBUFR AMV rows are filtered out unconditionally — so AMV dropout lives in the SATWND
loader and the conventional loader carries aircraft and scatterometer only.

`training=False` zeroes every random drop regardless of scope or stream state.

## Guards

- `fix_time_parallel_rng` defaults **True**. Sample scope depends on it.
- `train.py setup()` raises if `nnja_dropout_scope="sample"` runs with `time_parallel > 1`
  and the flag off, rather than letting the ranks disagree silently.
