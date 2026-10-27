# Examples

The scripts below live in the repository and are included here verbatim.

## Minimal training run

`examples/training/minimal.sh` runs the single-GPU smoke recipe. It is minimal in
duration, not in data requirements: it expects the prepared state Zarr and the NNJA
observation archive used by the selected recipe.

```bash title="examples/training/minimal.sh"
--8<-- "examples/training/minimal.sh"
```

## Fine-tune from a checkpoint

`examples/fine_tuning/from_checkpoint.sh` initializes model weights from a checkpoint
with `--finetune_from`. Optimizer and iterator state are not restored, so give a
fine-tuning preset its own learning-rate schedule.

```bash title="examples/fine_tuning/from_checkpoint.sh"
--8<-- "examples/fine_tuning/from_checkpoint.sh"
```
