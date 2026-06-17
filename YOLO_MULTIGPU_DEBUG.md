# YOLOv7 Multi-GPU Training Debug Summary

## Issues Found and Fixed

### 1. **Deprecated Command** ❌ → ✅
- **Problem**: Using `torch.distributed.launch` (deprecated in PyTorch 2.0+)
- **Fix**: Use `torch.distributed.run` instead

### 2. **Device Assignment Conflict** ❌ → ✅
- **Problem**: Passing `--device 0,1,2,3` when using DDP launcher
- **Fix**: Remove `--device` flag - the launcher assigns devices automatically via `LOCAL_RANK` environment variable

### 3. **Invalid Argument** ❌ → ✅
- **Problem**: `--patience 20` is not supported by YOLOv7's train.py
- **Fix**: Remove this unsupported argument

### 4. **PyTorch 2.6+ Compatibility** ❌ → ✅
- **Problem**: `torch.load()` with `weights_only` parameter changed in PyTorch 2.6+
- **Fix**: Try loading with `weights_only=True` first, fall back to `weights_only=False` on exception
- **Edited**: train.py lines 71 and 87

### 5. **Missing LOCAL_RANK Environment Variable** ❌ → ✅
- **Problem**: `torch.distributed.run` sets `LOCAL_RANK` env var, but script wasn't reading it
- **Fix**: Added code to read `LOCAL_RANK` from environment if `--local_rank` not explicitly set
- **Edited**: train.py after argument parsing

### 6. **GPU Support (Potential)** ⚠️
- **Note**: PyTorch 2.7+ drops support for V100 (compute capability 7.0)
- **Current**: Using PyTorch 2.5.1 which still supports V100
- **Warning**: If you upgrade PyTorch beyond 2.5.x, multi-GPU on V100 will fail with CUDA kernel errors

## Corrected Training Command

### Multi-GPU Training (4 GPUs):
```bash
cd /home/users/choij32/Cosmos-H-Surgical/yolov7

python -m torch.distributed.run \
  --nproc_per_node=4 \
  train.py \
  --data data/cholectrack20.yaml \
  --cfg cfg/training/yolov7.yaml \
  --weights yolov7.pt \
  --img 640 \
  --batch-size 32 \
  --epochs 100 \
  --name cholectrack20_yolov7_4gpu \
  --hyp data/hyp.scratch.p5.yaml
```

### Single GPU (for comparison):
```bash
python train.py \
  --data data/cholectrack20.yaml \
  --cfg cfg/training/yolov7.yaml \
  --weights yolov7.pt \
  --img 640 \
  --batch-size 32 \
  --epochs 100 \
  --device 0 \
  --name cholectrack20_yolov7_1gpu \
  --hyp data/hyp.scratch.p5.yaml
```

## Key Differences Between Commands

| Aspect | Single GPU | Multi-GPU (4x) |
|--------|-----------|----------------|
| Command | `python train.py ...` | `python -m torch.distributed.run --nproc_per_node=4 train.py ...` |
| Device Flag | `--device 0` | *(removed - auto-assigned)* |
| Batch Size | 32 (per GPU) | 32 (total, distributed) |
| Rank Management | Manual | Automatic via env vars |

## Files Modified

1. **train.py**:
   - Lines 71-78: Fixed `torch.load()` call for wandb_id with weights_only handling
   - Lines 83-90: Fixed `torch.load()` call for model checkpoint with weights_only handling
   - After line 576: Added `LOCAL_RANK` environment variable reading

## Environment Requirements

```
PyTorch: 2.5.1 (cu121)
CUDA: 12.1
GPUs: V100 (Tesla V100-DGXS-32GB)
Python: 3.13
```

## Troubleshooting

### If you get "CUDA error: no kernel image is available"
- **Cause**: PyTorch version too new for V100
- **Solution**: Keep PyTorch ≤ 2.5.x

### If you get "Weights only load failed"
- **Cause**: PyTorch 2.6+ changed default behavior
- **Solution**: Already fixed in train.py (fallback to weights_only=False)

### If processes fail to initialize distributed group
- **Cause**: LOCAL_RANK environment variable not being read
- **Solution**: Already fixed in train.py

## Testing Steps

1. **Test single GPU first**:
   ```bash
   cd yolov7
   python train.py --data data/cholectrack20.yaml --cfg cfg/training/yolov7.yaml \
     --weights yolov7.pt --img 640 --batch-size 32 --epochs 2 --device 0 \
     --name test_single_gpu --hyp data/hyp.scratch.p5.yaml
   ```

2. **Test multi-GPU**:
   ```bash
   cd yolov7
   python -m torch.distributed.run --nproc_per_node=4 train.py \
     --data data/cholectrack20.yaml --cfg cfg/training/yolov7.yaml \
     --weights yolov7.pt --img 640 --batch-size 32 --epochs 2 \
     --name test_4gpu --hyp data/hyp.scratch.p5.yaml
   ```

## Notes

- The batch-size remains 32 total in multi-GPU mode (distributed across 4 GPUs = 8 per GPU)
- Adjust batch-size if needed, but ensure it's divisible by `nproc_per_node`
- Use `NCCL_DEBUG=INFO` environment variable for debugging distributed issues
- For best performance, keep batch-size high but monitor GPU memory
