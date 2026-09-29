
    integration/  # integration tests that require network and/or dataset access

## Running Tests

### Single Process Tests
Most unit tests can be run with standard pytest:
```bash
pytest tests/unit/
```

### Distributed Tests
Some tests require multiple GPUs and need to be run with `torchrun`:
```bash
# Run specific test with 2 GPUs
torchrun --nproc_per_node=2 -m pytest tests/unit

# Run all distributed tests
torchrun --nproc_per_node=2 -m pytest tests/unit
```
