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

## Command-line entry points

`healda-train` and `healda-inference-local` are the installed console scripts.
Run either with `--help` for the authoritative option list.

::: healda.cli.train
    options:
      members: false

::: healda.cli.inference_local
    options:
      members: false
