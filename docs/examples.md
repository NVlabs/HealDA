# Examples

The scripts below live in the repository and are included here verbatim.

## Minimal training run

`examples/training/minimal.sh` runs `debug-single-gpu`, a quick check: the HealDA v2
model at reduced depth and time length 1, on one GPU. It needs
the ERA5 and NNJA archives prepared.

```bash title="examples/training/minimal.sh"
--8<-- "examples/training/minimal.sh"
```

## Launch from a preset file

Presets are YAML files in `src/healda/configs/loops/`, each extending another and listing
the fields it changes. `--config` takes a preset name or a file; a file may extend any
shipped preset. `--name` names the run directory and defaults to the preset or file
name. Floats need a decimal point and a signed exponent (`2.0e-4`, `1.0e+5`).

```yaml
# my-debug.yaml
extends: debug-single-gpu
loop:
  steps_per_tick: 10
```

```bash
torchrun --standalone --nproc-per-node=1 -m healda.cli.train \
  --config my-debug.yaml --output_dir ./training-runs --verbose
```

`--verbose` prints the resolved loop with every field, in the same YAML form.

## Fine-tune from a checkpoint

`examples/fine_tuning/from_checkpoint.sh` fine-tunes the v2 checkpoint with
`--finetune_from`. Optimizer and iterator state are not restored, so give a
fine-tuning preset its own learning-rate schedule.

```bash title="examples/fine_tuning/from_checkpoint.sh"
--8<-- "examples/fine_tuning/from_checkpoint.sh"
```
