# Earth2Studio wrapper for HealDA v2 (0.25° lat/lon recipe) — design

Status: DRAFT, pending approval. Scope agreed so far: approach A, observations
limited to GPS-RO and SATWND for the first version.

## Goal

Expose the `v2-nnja-latlon-final` HealDA checkpoint as an Earth2Studio
`AssimilationModel` (`earth2studio.models.da`) with `healda` installed as a package
dependency, so an Earth2Studio user can fetch NNJA observations with Earth2Studio
data sources and receive a 0.25° global analysis as an `xr.DataArray`.

## Context (verified in code)

- Earth2Studio `main` already ships the v1 wrapper `earth2studio/models/da/healda.py`
  (physicsnemo backbone, HPX level 6, UFS observations). Its protocol
  (`AssimilationModel`: `__call__(*dataframes) -> xr.DataArray`,
  `create_generator`, `input_coords` as `FrameSchema`s, `output_coords`,
  `load_default_package`, `load_model`) and its test/docs layout are the template.
- The v2 lat/lon model is observation-only: `Hpx256LatlonModel.forward` discards
  `hidden_states` and rebuilds its condition from ERA5 statics + calendar + geometry.
  It is deterministic (no sampler); output is `[b, 104, T, 721, 1440]` in normalized
  model space.
- Reconstruction needs `loop.json` from the `.checkpoint` zip:
  `TrainingLoop.loads(loop_json).get_network()` builds `Hpx256LatlonModel`;
  `model.json` alone cannot. Weights are `net_state.pth`
  (`torch.load(weights_only=True)`).
- ERA5 statics are fetched anonymously from the public NCAR mirror
  (`s3://nsf-ncar-era5/e5.oper.invariant/197901`) into `healda.config.environment.CACHE_DIR`
  at model construction; no private data is needed for the network itself.
- Observations flow `Earth2Studio NNJAObsConv/NNJAObsSatwnd frames ->
  healda.observations.adapters.e2s_nnja.{gpsro_tables,satwnd_tables} ->
  NNJAConventionalLoader(gpsro_table_source=..., satwnd_table_source=...) ->
  TransformV2._process_obs -> TransformV2.device_transform -> UnifiedObservation`.
  Sensors absent from a batch are legal ("remain present with zero counts").
- Gaps: `TransformV2.transform()` requires a `state` frame (only `_process_obs` is
  obs-only); `NNJAConventionalLoader` always constructs a PrepBUFR reader against
  the NNJA archive root, with no flag to skip it.

## Approach A — thin Earth2Studio wrapper over a small HealDA inference API

### HealDA side (NVlabs/HealDA, mirrored to GitLab)

1. `healda/inference.py`
   - `load_analysis_model(checkpoint: str | Path, map_location="cpu") -> AnalysisModel`
     where `AnalysisModel` bundles `net` (eval, no grad), the `TrainingLoop` it was
     built from, the `TransformV2` configured as in training (sensors, hpx levels,
     `use_nnja_sat`, `features_v2`, `static_condition` via `fixed_static_condition`),
     and `BatchInfo`/`NormalizationStats` for `era5_104ch_latlon`.
   - `AnalysisModel.analyze(times, obs_tables) -> torch.Tensor` returning physical-space
     `[b, 104, T, 721, 1440]` (denormalize with target stats, then
     `state_transforms.to_physical_space`).
2. `TransformV2.transform_observations(times, frames)` — public obs-only entry that
   wraps `_process_obs` plus the calendar/timestamp encodings, returning what
   `device_transform` accepts without a `target`.
3. `NNJAConventionalLoader(include_prepbufr: bool = True)` threading through to
   `NNJAConvLoader`, so GPS-RO + SATWND can run without a PrepBUFR archive.
4. Unit tests for 1-3 with a tiny synthetic `TrainingLoop` and synthetic tables;
   no checkpoint required.

### Earth2Studio side (NVIDIA/earth2studio)

1. `earth2studio/models/da/healda_v2.py::HealDAv2(torch.nn.Module, AutoModelMixin)`
   - `input_coords() -> (gpsro_schema, satwnd_schema)`: `FrameSchema`s matching what
     `NNJAObsConv(["gps", "gps_refractivity"])` and `NNJAObsSatwnd(["u", "v"])` return.
   - `output_coords(input_coords, request_time=None) -> (CoordSystem,)`:
     `time`, `variable` (104 names mapped to Earth2Studio vocabulary), `lat` 721
     (90..-90), `lon` 1440 (0..360, endpoint False).
   - `load_default_package()`: `hf://nvidia/healda-v2@<commit>` (commit pinned once
     the repository contents are confirmed).
   - `load_model(package, time_tolerance=(-3h, +3h - 1s))`: resolves the `.checkpoint`
     file and calls `healda.inference.load_analysis_model`.
   - `__call__(gpsro_obs=None, satwnd_obs=None) -> xr.DataArray`: requires at least
     one frame with `attrs["request_time"]`; converts with `e2s_nnja`; runs
     `AnalysisModel.analyze`; returns `[time, variable, lat, lon]` on the model device
     (cupy on CUDA, numpy on CPU), NaN-filled if no observations survive QC.
   - `create_generator()`: same shape as v1.
2. Dependency extra `da-healda-v2 = ["healda @ git+https://github.com/NVlabs/HealDA@<tag>", "earth2grid", "cupy-cuda13x", "cudf-cu13>=26.2.0"]`,
   included in `all`; `OptionalDependencyFailure("da-healda-v2")`.
3. `test/models/da/test_da_healda_v2.py`: mock `AnalysisModel` (Phoo) for
   `__call__`/generator/coords/empty-obs/missing-request-time; `@pytest.mark.package`
   test loading the real package.
4. Registration: `da/__init__.py`, `docs/modules/models_da.md`,
   `docs/userguide/about/install_options.yml`, `CHANGELOG.md`.
5. Example `examples/05_data_assimilation/03_healda_v2.py`: fetch GPS-RO and SATWND for
   one cycle, run the model, compare `t2m`/`z500` against `NCAR_ERA5`.

### Data flow

```
NNJAObsConv(gps, gps_refractivity) ─┐
                                    ├─ e2s_nnja tables ─ NNJAConventionalLoader.sel_time
NNJAObsSatwnd(u, v) ────────────────┘        │
                                             ▼
                        TransformV2.transform_observations ─ device_transform
                                             │
                                             ▼  UnifiedObservation + calendar
                        Hpx256LatlonModel(net) ─ denormalize ─ to_physical_space
                                             │
                                             ▼
                        xr.DataArray[time, variable(E2S), lat 721, lon 1440]
```

### Error handling

- Both frames `None` -> `ValueError`; missing `request_time` -> `ValueError`
  (same messages as v1).
- Frames outside `time_tolerance` are dropped; zero surviving observations -> warning
  and NaN output (v1 behavior), never a crash.
- Checkpoint without `loop.json` or with `latlon_decode` unset -> `ValueError`
  naming the expected recipe.
- Missing `healda`/`earth2grid` -> `OptionalDependencyFailure("da-healda-v2")`.

### Testing

- HealDA: unit tests for the inference API and the PrepBUFR skip flag; run in the
  existing `pytest tests/unit`.
- Earth2Studio: `uv run pytest test/models/da/test_da_healda_v2.py -m "not package" -v`
  on CPU with mocks; package test on a GPU node (Eos) with HF access.
- End-to-end on Eos: the example script for one 2024 cycle; sanity-check `t2m`/`z500`
  MAE against ERA5 is finite and of the same order as the v1 example.

## Open questions

1. Contents of `hf://nvidia/healda-v2` (returns 401 without a token): does it ship a
   `.checkpoint` zip containing `loop.json` + `net_state.pth` + `model.json` +
   `batch_info.json`? The design assumes yes.
2. Whether `NVlabs/HealDA` will be public (or tagged) by the time the Earth2Studio extra
   needs to resolve; a private git dependency is uninstallable for external users.
3. Analysis quality with GPS-RO + SATWND only (no PrepBUFR, no radiances) is expected to
   be degraded relative to the `obsAll` recipe; the example must say so.
4. `time_tolerance`: the NNJA loaders use a (-3 h, +3 h) cycle window while the model's
   `obs_config.context_start/end` is (-21, 3); confirm which the wrapper exposes.
