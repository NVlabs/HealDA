# Examples

The scripts below live in the repository and are included here verbatim.

## Minimal training run

`examples/training/minimal.sh` runs `debug-single-gpu`, a quick check: the HealDA v2
model at reduced depth and time length 1, on one GPU. It needs
the ERA5 and NNJA archives prepared.

```bash title="examples/training/minimal.sh"
--8<-- "examples/training/minimal.sh"
```

## Fine-tune from a checkpoint

`examples/fine_tuning/from_checkpoint.sh` fine-tunes the v2 checkpoint with
`--finetune_from`. Optimizer and iterator state are not restored, so give a
fine-tuning preset its own learning-rate schedule.

```bash title="examples/fine_tuning/from_checkpoint.sh"
--8<-- "examples/fine_tuning/from_checkpoint.sh"
```
