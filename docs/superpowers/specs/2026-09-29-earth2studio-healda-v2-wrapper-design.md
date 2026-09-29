# Earth2Studio wrapper for HealDA v2 (0.25° lat/lon recipe) — design

Status: DRAFT v2, pending approval. Revised after three independent code reviews
(HealDA side, Earth2Studio side, packaging). Scope agreed: approach A, observations
limited to GPS-RO and SATWND for the first version.

## Goal

Expose the `v2-nnja-latlon-final` HealDA checkpoint as an Earth2Studio
`AssimilationModel` (`earth2studio.models.da`) with `healda` installed as a package
dependency, so an Earth2Studio user can fetch NNJA observations with Earth2Studio
data sources and receive a 0.25° global analysis as an `xr.DataArray`.

## Context (verified in code; file:line in the HealDA repo unless noted)

- Earth2Studio `main` already ships the v1 wrapper `earth2studio/models/da/healda.py`
  (physicsnemo backbone, HPX level 6, UFS observations). Its protocol
  (`AssimilationModel`: `__call__(*dataframes) -> xr.DataArray`, `create_generator`,
  `init_coords`, `input_coords` as `FrameSchema`s, `output_coords`, `to`,
  `load_default_package`, `load_model`) and its test/docs layout are the template.
- The Earth2Studio pieces this design needs — `NNJAObsSatwnd`, the
  `gps_refractivity` lexicon entry, and the `radius_curvature`/`geoid_undulation`
  columns — exist only on Earth2Studio `main` at or after PR #1151 (commit
  `d62b60c3`). They are absent from the local `pr-1055` branch.
- The v2 lat/lon model is observation-only: `Hpx256LatlonModel.forward` deletes
  `hidden_states` (`models/hpx256_latlon.py:270`) and builds its condition from ERA5
  statics + calendar + geometry. It is deterministic. Output is
  `[b, 104, T=8, 721, 1440]` in normalized model space; the analysis is the **last**
  frame (`cli/train.py:1166-1168`, causal temporal attention).
- Recipe geometry: `time_length=8`, `time_step=6 h` (`cli/train.py:1669-1670`);
  per-frame observation window `context_start=-3, context_end=+3` hours
  (`cli/train.py:1696`, `config/models.py:12-13`). One analysis at time `t` therefore
  needs frames `t-42h, …, t` and observations spanning `[t-45h, t+3h)` — about nine
  NCEP 6-hourly cycle files.
- Reconstruction: `TrainingLoop.loads(loop_json).get_network()` (`cli/train.py:1283,
  1349`) builds `Hpx256LatlonModel`; `Checkpoint.read_model(net=...)` must be given
  that net (with `net=None` it builds a bare DiT, `training/checkpoint.py:70-71`).
  Non-FSDP checkpoints are zips whose `net_state.pth` has no `module.`/`_orig_mod.`
  prefixes (`training/loop.py:668-669`). `_migrate_latlon_fields` handles older
  `loop.json` layouts.
- The training dataset feeds a dummy condition `zeros(b, 1, t, 1)`
  (`datasets/da/hourly_latlon_dataset.py:220`); `fixed_static_condition` is **not**
  part of this recipe and would require `UFS_LAND_DATA_ZARR`.
- Statics for the lat/lon model are fetched anonymously from
  `s3://nsf-ncar-era5/e5.oper.invariant/197901` into `~/.cache/healda`
  (`datasets/era5_statics.py:43,79-84`) inside `Hpx256LatlonModel.__init__`; no `.env`
  is needed for the network.
- Observation vocabulary: `NNJAConventionalLoader` emits GSI `conv-plevel` ids
  (`observations/loaders/combined.py:301-310`); only `CombinedObsLoader.sel_time`
  rebases them into the NNJA vocabulary (`combined.py:99-115, 346`) that the latlon
  `TransformV2` uses (`use_nnja_sat=True`, `datasets/da/transform.py:317-323`).
  Unrebased rows land in zero-count buckets silently.
- `TransformV2._process_obs` (`transform.py:472`) + `device_transform` build a
  `UnifiedObservation` without a `target`; sensors absent from a batch are legal
  (`transform.py:459-466`) provided `sensors` stays the training list
  `[*DEFAULT_SENSORS, "conv-plevel"]` (`datasets/da/tasks.py:241-243`).
- A missing PrepBUFR archive is a silent no-op (`nnja_conventional.py:438-440`) and
  GPS-RO/SATWND with `table_source` never touch `archive_root`
  (`nnja_gpsro.py:389-394`), so GPS-RO + SATWND-only inference needs no new loader
  flag; an explicit `include_prepbufr=False` is cleanliness only.
- Denormalization: `era5_104ch_latlon_stats.csv` (104 rows) and
  `state_transforms.to_physical_space` (tcw residual, `state_transforms.py:47-52,96`).
  `sst`/`sic` are masked-domain channels (`use_masked_domain_loss`,
  `cli/train.py:1725`) and are meaningless over land unless
  `state_transforms.restore_fill` (`:113`) is applied. All 104 channel names map to the
  Earth2Studio lexicon (`earth2studio/lexicon/arco.py`), including `w{lev}`,
  the 10 hPa level, `skt`, `stl1/2`, `swvl1/2`, `sd`, `tcw`, `lcc/mcc/hcc`.
- Hard runtime requirements: CUDA GPU and Triton (`triton` is imported unconditionally
  via `healda.models.dit`, `kernels/triton_pixel_attention.py:65`); flex-attention
  block masks are compiled. CuTe DSL and Transformer Engine degrade gracefully.
- Import footprint: `TrainingLoop` is defined in `cli/train.py`, which imports
  `matplotlib.pyplot` (line 18) and `cartopy` via `healda.utils.visualization`
  (line 64); `training/loop.py` imports `psutil` and `torch.utils.tensorboard` at
  module level. Only `wandb`, `duckdb`, `obstore` are lazy.
- Packaging (dry-run `uv lock` on scratch copies, uv 0.10.9): every HealDA version
  floor is satisfied by Earth2Studio's current lock (torch 2.13, triton 3.7.1,
  physicsnemo 2.2.0a0 git pin, xarray 2026.2, zarr 3.2.1, numba 0.64, ...); only
  `diffusers` and `duckdb` are new. But `earth2studio[all]` + `healda` **fails to
  resolve** because both declare a source for `earth2grid` (HealDA:
  `url = .../archive/main.tar.gz`; Earth2Studio: `git = ..., rev = 11dcf1b0`) — uv does
  honor a git dependency's own `[tool.uv.sources]`. Removing HealDA's `earth2grid`
  source entry makes the full `all` lock resolve cleanly (453 packages).
  HealDA's CSV package data ships in the wheel (hatchling `packages = ["src/healda"]`,
  files under `src/healda/**/normalizations/`).

## Approach A — thin Earth2Studio wrapper over a small HealDA inference API

### HealDA side (NVlabs/HealDA, mirrored to GitLab main via MR)

1. **`healda/inference.py`** (new, GPU-only at runtime)
   - `load_analysis_model(checkpoint: str | Path, device) -> AnalysisModel`:
     opens the `.checkpoint` zip with `Checkpoint`, `TrainingLoop.loads(loop.json)`,
     validates `latlon_decode` is set (else `ValueError` naming the expected recipe),
     `net = loop.get_network()`, `read_model(net=net, map_location="cpu")`, `.eval()`,
     `requires_grad_(False)`, `.to(device)`.
   - `AnalysisModel` bundles `net`, the `TrainingLoop`, a `TransformV2` configured
     exactly as `hourly_latlon_dataset.py:297-307` does (sensors = training list,
     `hpx_level`, `hpx_level_condition`, `features_v2`, `use_nnja_sat`,
     `attention_prepack`, `pixel_order`, `variable_config`), the observation loader
     built as `CombinedObsLoader(conventional=NNJAConventionalLoader(..., gpsro_table_source, satwnd_table_source), satellite=None)`
     so the vocabulary rebase runs, and `NormalizationStats` for `era5_104ch_latlon`.
   - `AnalysisModel.frame_times(t) -> pd.DatetimeIndex` = `t - 42h … t` step 6 h;
     `AnalysisModel.observation_window(t) -> (t - 45h, t + 3h)`.
   - `AnalysisModel.analyze(analysis_times, gpsro_tables, satwnd_tables) -> torch.Tensor`
     of physical-space `[b, 104, 721, 1440]` (last frame only): for each analysis
     time runs `loader.sel_time(frame_times)`, `TransformV2.transform_observations`,
     `device_transform`, forward with the kwargs `run_model_step` uses
     (`training/step.py:288-298`: dummy `condition=zeros(b,1,T,1)`,
     `noise_labels=zeros(b)`, `class_labels=empty(b,0)`, `second_of_day`,
     `day_of_year`, `timestamp`, `unified_obs`, `is_causal`), under bf16 autocast;
     then denormalize, `to_physical_space`, `restore_fill` for masked channels, and
     select `T=-1`.
2. **`TransformV2.transform_observations(times, frames)`** — public obs-only entry
   wrapping `_process_obs` plus the calendar/timestamp encodings, returning what
   `device_transform` accepts without a `target`.
3. **`NNJAConventionalLoader(include_prepbufr: bool = True)`** threaded to
   `NNJAConvLoader` (cleanliness; not a blocker).
4. **`pyproject.toml`: remove the `earth2grid` entry from `[tool.uv.sources]`**
   so `healda` can be a dependency of another uv project. This is a dependency-spec
   change and needs explicit approval; HealDA's own lock then resolves `earth2grid`
   from PyPI (`>=2025.11.1` is on PyPI) — to be verified with `uv lock --check` after
   the edit.
5. **Recommended follow-up, not required for v1**: move the `TrainingLoop` dataclass
   and `get_network` out of `cli/train.py` into a plotting-free module and make
   `psutil`/`tensorboard` imports lazy, so inference does not import matplotlib and
   cartopy. Until then those remain hard (but already-declared) dependencies.
6. Unit tests for 1-3 with a tiny synthetic `TrainingLoop` and synthetic tables,
   `load_era5_statics_latlon/hpx` stubbed, network on the meta device as in
   `tests/unit/test_latlon_decode_wiring.py:31-33`; no checkpoint or GPU required.

### Earth2Studio side (NVIDIA/earth2studio, branch off `main` ≥ #1151)

1. **`earth2studio/models/da/healda_v2.py::HealDAv2(torch.nn.Module, AutoModelMixin)`**,
   decorated `@check_optional_dependencies()`, with a `device_buffer` and `device`
   property as in v1, and a `Badges` block in the class docstring.
   - `init_coords() -> None`.
   - `input_coords() -> (gpsro_schema, satwnd_schema)`: `FrameSchema`s matching
     `NNJAObsConv(["gps", "gps_refractivity"])` and `NNJAObsSatwnd(["u", "v"])` on
     `main` ≥ #1151 (fields: `time, lat, lon, observation, variable, elev, pres,
     quality, radius_curvature, geoid_undulation` / `time, lat, lon, observation,
     variable, pres, quality, satellite_id, subset, wind_method, wind_method_local,
     height_method, satellite_za`).
   - `output_coords(input_coords, request_time=None) -> (CoordSystem,)`: `time`,
     `variable` (104 Earth2Studio names, extending v1's `_CHANNEL_TO_E2S`), `lat` 721
     (90..-90), `lon` 1440 (0..360, endpoint False).
   - `load_default_package()`: `Package("hf://nvidia/healda-v2@<commit>", cache_options={"same_names": True})`;
     document `EARTH2STUDIO_PACKAGE_TIMEOUT` for the multi-GB checkpoint.
   - `load_model(package, device)`: resolves the `.checkpoint` file and calls
     `healda.inference.load_analysis_model`.
   - `__call__(gpsro_obs=None, satwnd_obs=None) -> xr.DataArray`: requires at least
     one frame carrying `attrs["request_time"]` (set by `fetch_dataframe`; direct
     source calls do not set it); converts cudf to pandas; adapts with
     `healda.observations.adapters.e2s_nnja.{gpsro_tables, satwnd_tables}`; runs
     `AnalysisModel.analyze`; returns `[time, variable, lat, lon]` on the model device
     (cupy on CUDA, numpy on CPU), NaN-filled if no observations survive.
   - `create_generator()`: same shape as v1.
2. **Dependencies**: `[tool.uv.sources] healda = { git = "https://github.com/NVlabs/HealDA.git", rev = "<sha>" }`
   and extra `da-healda-v2 = ["healda", "earth2grid", "cupy-cuda13x", "cudf-cu13>=26.2.0"]`
   (plain names only, per repo convention). Inclusion in `all` is contingent on HealDA
   item 4; until then `da-healda-v2` stays out of `all` with a comment.
3. **`test/models/da/test_da_healda_v2.py`**: Phoo `AnalysisModel` for
   `__call__`/generator/coords/empty-obs/missing-`request_time`/cudf-input;
   `@pytest.mark.package` test loading the real package on a GPU; add
   `test_da_healda_v2.py: ["da-healda-v2"]` to the dependency map in
   `test/conftest.py:116`.
4. **Registration**: `earth2studio/models/da/__init__.py`, `docs/modules/models_da.md`
   autosummary (alphabetical), `docs/userguide/about/install_options.yml` entry with
   `api_refs` and a `preinstall`/`warnings` block for the git-pinned `healda`,
   `docs/userguide/components/data_assimilation.md` snippet, `CHANGELOG.md`
   "### Added".
5. **Example** `examples/05_data_assimilation/03_healda_v2.py`: `fetch_dataframe`
   with `time_tolerance=(-45 h, +3 h - 1 s)` for GPS-RO and SATWND, run the model for
   one 2024 analysis time, compare `t2m`/`z500` against `NCAR_ERA5`, and state that
   GPS-RO + SATWND alone is a degraded subset of the `obsAll` recipe.

### Data flow

```
fetch_dataframe(NNJAObsConv(gps, gps_refractivity), t, tol=(-45h,+3h)) ─┐   attrs.request_time
fetch_dataframe(NNJAObsSatwnd(u, v),               t, tol=(-45h,+3h)) ─┤
                                                                        ▼ cudf -> pandas
                              e2s_nnja.gpsro_tables / satwnd_tables  {cycle: pa.Table}
                                                                        │
                       CombinedObsLoader(NNJAConventionalLoader(table sources)).sel_time(8 frame times)
                                                                        │  GSI ids -> NNJA vocabulary
                              TransformV2.transform_observations ─ device_transform
                                                                        │  UnifiedObservation + calendar
                  Hpx256LatlonModel(net)(dummy condition, ...) -> [b,104,8,721,1440]
                                                                        │  denormalize, to_physical_space,
                                                                        │  restore_fill, select T=-1
                              xr.DataArray[time, variable(E2S), lat 721, lon 1440]
```

### Error handling

- Both frames `None` -> `ValueError`; missing `request_time` -> `ValueError`
  (same messages as v1).
- Observations outside `[t-45h, t+3h)` are dropped by the loaders; zero surviving
  observations -> warning and NaN output (v1 behavior), never a crash.
- Checkpoint without `loop.json`, or with `latlon_decode` unset -> `ValueError`
  naming the expected recipe.
- No CUDA device -> `RuntimeError` at `load_model` (the model cannot run on CPU).
- Missing `healda`/`earth2grid` -> `OptionalDependencyFailure("da-healda-v2")`.

### Testing

- HealDA: unit tests for items 1-3 (meta-device network, stubbed statics, synthetic
  tables) in `pytest tests/unit`; `uv lock --check` after item 4.
- Earth2Studio: `uv run pytest test/models/da/test_da_healda_v2.py -m "not package" -v`
  on CPU with mocks; `--package` test on an Eos GPU node with HF access.
- End-to-end on Eos: the example for one 2024 cycle; check `t2m`/`z500` MAE against
  ERA5 is finite and of the same order as the v1 example; record peak GPU memory for a
  single-GPU T=8 forward of the 5B backbone (training used `time_parallel=8`).

## Open questions / decisions needed

1. **HF package contents** (`hf://nvidia/healda-v2` returns 401 without a token; no
   token in this environment): does it ship a `.checkpoint` zip containing
   `loop.json` + `net_state.pth` + `model.json` + `batch_info.json`, and which
   `loop.json` layout (pre/post `_migrate_latlon_fields`)?
2. **Approval to remove HealDA's `earth2grid` `[tool.uv.sources]` entry** (HealDA
   item 4). Without it `da-healda-v2` cannot join `earth2studio[all]`.
3. **Whether to do the `TrainingLoop` refactor now** (HealDA item 5) or accept
   matplotlib/cartopy as inference imports for v1.
4. **Single-GPU feasibility** of T=8 inference for the 5B backbone (memory), and
   whether `time_length < 8` is legal at inference (`time_length` is baked into
   `model_config`, `cli/train.py:1428`) — to be measured on Eos.
5. `NVlabs/HealDA` visibility/tag for the git pin: a private repo makes the extra
   uninstallable for external users.
