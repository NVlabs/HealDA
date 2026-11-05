# API reference

The entry points HealDA supports for use from other code. Everything else under
`healda` may change without notice.

## Inference

::: healda.inference
    options:
      members:
        - load_da_model
        - DAModel

## Earth2Studio observation adapter

::: healda.observations.adapters.e2s_nnja
    options:
      members:
        - analysis_tables
        - analysis_tables_long

## Observation table schemas

::: healda.observations.schema
    options:
      members:
        - table_schema
        - validate

## Training presets

One YAML file per preset in `src/healda/configs/loops/`, each extending another, read by
`healda-train --config`; `--verbose` prints the resolved loop in the same form.

::: healda.training.presets
    options:
      members:
        - load_preset
        - load_preset_dir
        - to_yaml

## Command-line entry points

`healda-train` and `healda-inference-local` are the installed console scripts.
Run either with `--help` for the authoritative option list.

::: healda.cli.train
    options:
      members: false

::: healda.cli.inference_local
    options:
      members: false
