# Notes on PyTorch FSDP and DTensor for 3D Parallelism

## Summary

**Our Implementation:**
- ✅ **3D parallelism**: Data Parallel (DP), Time Parallel (TP), Space Parallel (SP)
- ✅ FSDP over the DP dimension for hybrid-sharded data parallelism (HSDP)
- ✅ Manual time/space parallelism using process groups and all-to-all collectives
- ✅ Regular tensor interfaces between the three strategies

**Why We Didn't Use DTensor:**
- FSDP does not accept DTensor inputs (runtime type mismatch errors)
- DTensor's redistribute operations don't map cleanly to our resharding patterns
- The process group approach gave us direct control over communication

**Key Takeaway:** For combining FSDP with time and space parallelism, manual implementation with process groups and regular tensors proved more practical than attempting to use DTensor's abstractions.

---

## Our 3D Parallelism Strategy (DP + Time + Space)

For training video diffusion transformers, we combine three parallelism dimensions:

1. **Data parallel (DP)** — Replicate/shard the batch across DP ranks; FSDP shards parameters and optimizer state.
2. **Time parallel (TP)** — Shard the time/sequence dimension across TP ranks; spatial attention uses time-sharded layout.
3. **Space parallel (SP)** — Shard the spatial dimension across SP ranks use ring attention; temporal attention uses time-contiguous layout with spatial sharded across (TP×SP).

The model alternates between temporal and spatial attention. Temporal attention uses a **time-contiguous** layout (each rank holds `[batch, time, spatial / SP / TP, channels]`). Spatial attention uses a **time-sharded** layout (each rank holds `[batch, time / TP, space/SP, channels]`). We use manual all-to-all resharding between the two layouts over the (TP, SP) process groups.

```python
# 3D mesh: [Data Parallel, Time Parallel, Space Parallel]
mesh_3d = init_device_mesh(
    "cuda", (dp_size, tp_size, sp_size),
    mesh_dim_names=["data", "time", "space"]
)
# FSDP shards over the data dimension
dp_mesh = mesh_3d["data"]
# Process groups over (TP, SP) for resharding between time-contiguous and time-sharded layouts
time_space_group = mesh_3d.get_group(["time", "space"])

class TransformerBlock(nn.Module):
    def __init__(self):
        self.temporal_attn = Attention()
        self.spatial_attn = Attention()
        self._time_space_group = None

    def set_time_space_group(self, group):
        """Set the (TP, SP) process group for layout resharding"""
        self._time_space_group = group

    def forward(self, x):
        # x: time-contiguous [batch, time, spatial/(SP*TP), channels] on each rank

        # Temporal attention on time-contiguous layout
        x = self.temporal_attn(x)

        # Reshard: time-contiguous -> time-sharded for spatial attention
        x = shard_to_spatial(x, self._time_space_group)  # all-to-all
        x = self.spatial_attn(x)

        # Reshard back: time-sharded -> time-contiguous for next layer
        x = shard_to_sequence(x, self._time_space_group)  # all-to-all

        return x

# FSDP over data dimension; time/space handled by resharding
block = TransformerBlock()
block.set_time_space_group(time_space_group)
fully_shard(block, mesh=mesh_3d)

# Forward: each rank has time-contiguous local input
x = torch.randn(batch, time, local_spatial, channels, device=device)
y = block(x)  # Regular tensor in and out
```

The resharding functions use explicit all-to-all collectives over the (TP, SP) process group to switch between time-contiguous and time-sharded layouts:

```python
from torch.distributed.nn.functional import all_to_all_single
import einops

def shard_to_spatial(tensor, group):
    """Time-contiguous -> time-sharded: [b, time, spatial/(SP*TP), c] -> [b, time/TP, space/SP, c]"""
    n = dist.get_world_size(group)
    tensor = einops.rearrange(tensor, "b t (n x) c -> n b t x c", n=n)
    tensor = tensor.contiguous()
    output = torch.empty_like(tensor)
    output = all_to_all_single(output, tensor, group=group)
    output = einops.rearrange(output, "n b t x c -> b (n t) x c")
    return output

def shard_to_sequence(tensor, group):
    """Time-sharded -> time-contiguous: [b, time/TP, space/SP, c] -> [b, time, spatial/(SP*TP), c]"""
    n = dist.get_world_size(group)
    tensor = einops.rearrange(tensor, "b (n t) x c -> n b t x c", n=n)
    tensor = tensor.contiguous()
    output = torch.empty_like(tensor)
    output = all_to_all_single(output, tensor, group=group)
    output = einops.rearrange(output, "n b t x c -> b t (n x) c")
    return output
```

This gives 3D parallelism: FSDP over DP, and manual resharding over (TP, SP) between time-contiguous and time-sharded layouts for temporal and spatial attention, all with regular tensors.


## Why We Didn't Use DTensor for Time/Space Parallelism

After implementing our manual approach, we explored whether DTensor could simplify the code by providing a higher-level abstraction for the resharding operations. However, we encountered a fundamental compatibility issue: FSDP does not accept DTensor inputs.

### The Type Mismatch Problem

When implementing 3D parallelism with FSDP and DTensor, we discovered that FSDP manages parameters by converting them to DTensors for storage, but during forward passes, it all-gathers and converts them back to regular `torch.Tensor` objects before invoking the module's forward method. This design allows FSDP to maintain compatibility with PyTorch's optimizer infrastructure. However, if you pass DTensor inputs to an FSDP-wrapped module, the forward computation encounters mixed types—DTensor inputs with regular Tensor parameters—which triggers a runtime error: `"aten.addmm.default: got mixed torch.Tensor and DTensor"`.

```python
# This pattern produces a type mismatch error with FSDP
from torch.distributed.tensor import distribute_tensor, Replicate, Shard
from torch.distributed.fsdp import fully_shard

mesh = init_device_mesh("cuda", (2, 2, 2), mesh_dim_names=["dp", "time", "space"])
linear = nn.Linear(16, 8).to(device)
fully_shard(linear, mesh=mesh)  # Parameters become DTensors internally

# Create DTensor input (sharded on time dimension)
x = torch.randn(4, 8, 16, device=device)
x_dtensor = distribute_tensor(x, mesh, [Replicate(), Shard(1), Replicate()])

# Runtime error: mixed DTensor and Tensor
y = linear(x_dtensor)  # ❌ Type mismatch error
```

This limitation wasn't immediately apparent from the documentation, and we discovered it through experimentation. Since FSDP was essential for our data parallelism strategy, this meant DTensor couldn't be used for our activation resharding.

### DTensor Redistribute Overhead

Even setting aside the FSDP compatibility issue, we found that using DTensor for our resharding patterns would add unnecessary abstraction. Our resharding is fundamentally an all-to-all collective that transforms between time-contiguous and time-sharded layouts. Using DTensor's `redistribute()` operations would wrap this in additional API calls without providing clear benefits:

```python
# DTensor approach would look like:
x = distribute_tensor(x, mesh, [Replicate(), Replicate(), Shard(2)])  # Time-contiguous (spatial sharded)
temporal_output = temporal_attention(x)

# Redistribute for spatial attention (time-contiguous -> time-sharded)
x_spatial = x.redistribute(mesh, [Replicate(), Replicate(), Shard(2)])
spatial_output = spatial_attention(x_spatial)

# This wraps our all-to-all in additional API layers
```

Our direct all-to-all implementation is more explicit about what's happening—we're performing a specific data layout transformation between time-contiguous and time-sharded—and gives us precise control over the einops rearrangements. The DTensor abstraction would add another layer without making the code clearer for our specific use case.

## Where DTensor Does Work Well

DTensor does work well for its primary use case: tensor parallelism with static weight sharding. When a linear layer's weight matrix is large, DTensor provides a clean abstraction for column-wise or row-wise parallelism. The `torch.distributed.tensor.parallel` module handles the collective operations automatically, and since weight matrices maintain their sharding pattern throughout training, DTensor's static approach fits naturally.

```python
# DTensor for static tensor parallelism works well:
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel

tp_mesh = init_device_mesh("cuda", [4])  # 4-way tensor parallel

# Column-wise parallel: shard output dimension
# Weight: [out_features/4, in_features] on each device
parallelize_module(
    model.layers,
    tp_mesh,
    {"weight": ColwiseParallel(), "bias": ColwiseParallel()}
)

# DTensor handles the all-reduce after matmul automatically
```

The challenge arises when trying to extend DTensor to activation sharding with dynamic patterns, or composing it with FSDP for data parallelism. For our use case, we found it more practical to keep DTensor focused on what it does well—static weight sharding—and handle dynamic activation resharding separately with process groups.

## Lessons Learned

Our experience implementing 3D parallelism (DP + time + space) led to a few key insights:

1. **Start with process groups for dynamic resharding**: For patterns that require changing data layouts between layers (time-contiguous ↔ time-sharded), process groups with all-to-all give you direct control and compose cleanly with FSDP.

2. **FSDP requires regular tensor inputs**: This isn't prominently documented, but FSDP's internal parameter management converts DTensors to regular tensors during forward passes, so passing DTensor inputs causes type mismatches.

3. **DTensor works for static patterns**: If you need tensor parallelism for large weight matrices (column/row parallel layers), DTensor provides a clean abstraction. Just be aware it doesn't compose with FSDP for activation inputs.

4. **Separation is practical**: Rather than trying to unify everything under one abstraction, keeping data parallelism (FSDP) and time/space parallelism (process groups) separate with regular tensor interfaces proved more maintainable.

Our implementation has been stable in production. The explicit all-to-all operations make debugging straightforward, and the separation of concerns means we can test and optimize each of the three dimensions (DP, time, space) independently. While a more unified API would be conceptually appealing, the current approach works well for our use case.

