# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import torch
from torch.distributed.checkpoint.stateful import Stateful
import torch.distributed.checkpoint.state_dict as S
from torch.distributed.checkpoint.state_dict import StateDictOptions
import torch.distributed.checkpoint as dcp
import json
from healda.training import checkpoint
import shutil
import torch.distributed as torch_dist


class AppState(Stateful):
    """This is a useful wrapper for checkpointing the Application State. Since this object is compliant
    with the Stateful protocol, DCP will automatically call state_dict/load_stat_dict as needed in the
    dcp.save/load APIs.

    Note: We take advantage of this wrapper to hande calling distributed state dict methods on the model
    and optimizer.

    from: https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
    """

    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        out = {}
        out["model"] = S.get_model_state_dict(self.model)
        if self.optimizer is not None:
            out["optimizer"] = S.get_optimizer_state_dict(self.model, self.optimizer)
        return out

    def load_state_dict(self, state_dict):
        # sets our state dicts on the model and optimizer, now that we've loaded
        S.set_model_state_dict(self.model, state_dict["model"])
        if self.optimizer is not None:
            S.set_optimizer_state_dict(
                self.model, self.optimizer, state_dict["optimizer"]
            )


def is_valid(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    return os.path.exists(os.path.join(path, ".metadata"))


def _default_training_metadata() -> dict:
    return {
        "epoch_idx": 0,
        "samples_processed_this_epoch_per_rank": 0,
        "wandb_id": None,
    }


def load_sidecar_metadata(checkpoint_dir: str) -> dict:
    """Optional metadata.json next to a DCP checkpoint (training loop state)."""
    metadata = _default_training_metadata()
    metadata_path = os.path.join(checkpoint_dir, "metadata.json")
    if not os.path.isfile(metadata_path):
        return metadata
    try:
        with open(metadata_path) as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(
            f"Checkpoint metadata at {metadata_path} is unreadable: {e}"
        ) from e
    if not isinstance(raw, dict):
        raise ValueError(
            f"Checkpoint metadata at {metadata_path} must be a JSON object, got {type(raw)!r}"
        )
    for key in metadata:
        if key in raw:
            metadata[key] = raw[key]
    return metadata


_metadata_group = None


def _get_metadata_group():
    # DCP's planning collectives (scatter_object_list / gather over the plan
    # metadata) run on the process group's device. On NCCL they share the GPU
    # with the checkpoint memory spike and intermittently fail with a generic
    # "unhandled cuda error". Route those metadata-only collectives over Gloo
    # (CPU) instead; the tensor payload is written to disk by the storage
    # writer and never crosses this group, so there is no throughput cost.
    global _metadata_group
    if _metadata_group is None:
        _metadata_group = torch_dist.new_group(backend="gloo")
    return _metadata_group


def save(path, model, optimizer=None, metadata=None):
    tmppath = path + ".tmp"

    # Clean up possibly stale .tmp from a previously interrupted save at the same nimg
    if torch_dist.get_rank() == 0 and os.path.exists(tmppath):
        shutil.rmtree(tmppath)
    torch_dist.barrier()

    dcp.save(
        {"app": AppState(model, optimizer)},
        checkpoint_id=tmppath,
        process_group=_get_metadata_group(),
    )

    # Rank 0 writes metadata and performs the atomic rename
    if torch_dist.get_rank() == 0:
        if metadata is not None:
            with open(os.path.join(tmppath, "metadata.json"), "w") as f:
                json.dump(metadata, f)
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.rename(tmppath, path)
    torch_dist.barrier()


def load(path, model, optimizer=None, require_all=True):
    """
    Load a checkpoint into an FSDP model.

    Supports both:
    - FSDP distributed checkpoints (directories)
    - Single process checkpoints (files/zip files)

    Args:
        path: Path to checkpoint (directory for FSDP, file for single process)
        model: FSDP-wrapped model to load into
        optimizer: Optional optimizer to load state for
        require_all: Whether to require all parameters to match (strict loading)

    Returns:
        Dictionary containing metadata from the checkpoint
    """
    if os.path.isdir(path):
        # FSDP distributed checkpoint
        dcp.load(
            {"app": AppState(model, optimizer)},
            checkpoint_id=path,
            process_group=_get_metadata_group(),
        )
        return load_sidecar_metadata(path)
    else:
        # Single process checkpoint - all ranks read and FSDP distributes
        with checkpoint.Checkpoint(path, "r") as state:
            # Load the full state dict (all ranks) - keep on CPU to avoid OOM
            with state.open("net_state.pth", "r") as f:
                net_state = torch.load(f, weights_only=True, map_location="cpu")

            # Use FSDP's set_model_state_dict to properly distribute it
            S.set_model_state_dict(
                model,
                model_state_dict=net_state,
                options=StateDictOptions(
                    full_state_dict=True,  # Indicate this is a full state dict
                    strict=require_all,
                ),
            )

            # Load optimizer state if available and requested
            if optimizer is not None:
                try:
                    with state.open("optimizer_state.pth", "r") as f:
                        optimizer_state = torch.load(
                            f, weights_only=True, map_location="cpu"
                        )
                        S.set_optimizer_state_dict(
                            model,
                            optimizer,
                            optim_state_dict=optimizer_state,
                            options=StateDictOptions(
                                full_state_dict=True, strict=require_all
                            ),
                        )
                except (KeyError, FileNotFoundError):
                    # Optimizer state not available in single process checkpoint
                    pass

            metadata = _default_training_metadata()
            try:
                with state.open("iterator_state.json") as f:
                    iterator_state = json.loads(f.read())
                if isinstance(iterator_state, dict):
                    metadata["epoch_idx"] = iterator_state.get("epoch_idx", 0)
                    metadata["samples_processed_this_epoch_per_rank"] = (
                        iterator_state.get("samples_processed_this_epoch_per_rank", 0)
                    )
            except (KeyError, FileNotFoundError):
                pass

            try:
                with state.open("loop.json") as f:
                    loop_fields = json.loads(f.read())
                if isinstance(loop_fields, dict):
                    metadata["wandb_id"] = loop_fields.get("wandb_id")
            except (KeyError, FileNotFoundError):
                pass

            return metadata
