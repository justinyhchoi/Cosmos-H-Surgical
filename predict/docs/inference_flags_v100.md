# Inference Flags Reference (V100 Focus)

This guide documents the flags available when running:

## How the CLI Is Structured

The example runner in examples/inference.py combines three groups of arguments:

1. Runner arguments
- -i, --input-files: one or more json/jsonl/yaml sample files

2. Setup arguments (model/runtime setup)
- Provided via CLI
- Applied once when loading the model

3. Inference override arguments (per-sample generation behavior)
- Can come from CLI and/or input json
- CLI values override file values
- CPU RAM up: shifts memory pressure from GPU to host

## Runner Flag

| Flag | Type | Default | Meaning | Memory impact |
|---|---|---|---|---|
| -i, --input-files | list[path] | required (no default) | Input parameter files (.json, .jsonl, .yaml). | VRAM neutral |

## Setup Flags (examples/inference.py)

| Flag | Type | Default | Meaning | Memory impact |
|---|---|---|---|---|
| -o, --output-dir | path | required | Output folder. | VRAM neutral |
| --model | string | 2B/base/post-trained default | Model key from registry. | Can be VRAM up/down depending on model size/variant |
| --checkpoint-path | string/path | auto from model | Override checkpoint location. | VRAM neutral by itself |
| --experiment | string | auto from model | Override experiment config name. | Depends on selected config |
| --config-file | string | auto | Config python module path for model assembly. | Depends on selected config |
| --context-parallel-size | int | WORLD_SIZE (or 1) | Number of GPUs for context parallel inference. | Per-GPU VRAM down; total system GPU memory up |
| --offload-diffusion-model | bool | false | Offload DiT/diffusion model to CPU when not actively running. | VRAM down, CPU RAM up, slower |
| --offload-tokenizer | bool | false | Offload tokenizer module to CPU. | VRAM down (small/moderate), CPU RAM up |
| --offload-text-encoder | bool | false | Offload Reason1/Qwen text encoder to CPU. | VRAM down (moderate), CPU RAM up |
| --disable-guardrails | bool | true | Disable text/video guardrail models. | If false (enable guardrails): extra memory required |
| --offload-guardrail-models | bool | true | Keep guardrail models on CPU when enabled. | VRAM down, CPU RAM up |
| --keep-going | bool | true | Continue batch after failures. | VRAM neutral |
| --profile | bool | false | Enable profiler outputs. | Slight VRAM/CPU overhead possible |
| --skip-existing-output | bool | false | Skip samples whose outputs already exist. | VRAM neutral |

### Notes for V100

- For low VRAM pressure on V100, first try:
  - --offload-diffusion-model
  - --offload-text-encoder
  - --offload-tokenizer
- If using multiple V100s, increase --context-parallel-size and launch with torchrun --nproc_per_node=N.
- Offloading trades memory for latency; expect slower throughput.

## Inference Override Flags

These correspond to fields in input json, but can be overridden from CLI.

## image2world vs video2world

At inference time, the practical difference is how many conditioning frames are provided to the model and what kind of temporal continuity you preserve from the input:

- image2world
  - Uses 1 conditional frame (single-image style conditioning).
  - Works with an image input, and can also accept a video path but only uses image-like conditioning behavior.
  - Good when you want to start from one still frame and generate future motion.
  - Usually the lower-memory option versus full video conditioning.

- video2world
  - Uses 2 conditional frames (short temporal conditioning window).
  - Uses the beginning/end of input video context (depending on preprocessing path) to preserve temporal continuity.
  - Better when you already have motion context and want continuation.
  - Usually slightly higher compute/memory pressure than image2world.

In this code path, these map to different internal num_input_frames values:
- text2world = 0
- image2world = 1
- video2world = 2

## Resolution Guidance

- Default resolution behavior
  - --resolution default is none.
  - none means use the model trained/default resolution from config.
  - For the base Cosmos-H-Surgical examples, that effectively maps to the 720p family used by the model pipeline.

- Safe choices that are least likely to break runs
  - Keep --resolution=none unless you are actively testing limits.
  - Use model-aligned sizes first: 704,1280 (commonly used for 720p pipeline alignment).
  - Lower-memory fallback: 432,768.

- What can break
  - Very large H,W combinations can OOM on V100.
  - Non-aligned shapes may get internally adjusted or fail in some paths, depending on tokenizer/model constraints.
  - If you test custom resolutions, increase gradually and keep num_steps fixed while probing limits.

| Flag | Type | Default | Meaning | Memory impact |
|---|---|---|---|---|
| --inference-type | text2world\|image2world\|video2world | required (no default) | Generation mode. | video2world can use more compute/memory than text2world |
| --input-path | path | none (required for image2world/video2world) | Input image/video path (required for image2world/video2world). | VRAM neutral (I/O side) |
| --prompt | string | none (unless loaded from prompt_path/input file) | Text prompt. | VRAM neutral |
| --prompt-path | path | none | Load prompt text from file. | VRAM neutral |
| --negative-prompt | string | built-in long default negative prompt | Negative prompt for CFG. | VRAM neutral/minor |
| --seed | int | 0 | RNG seed. | VRAM neutral |
| --guidance | int (0-7) | 7 | Prompt adherence strength (CFG). | Usually compute up; VRAM usually similar/slightly up |
| --resolution | string (H,W or none) | none (use model default resolution) | Output resolution override. | VRAM up strongly with larger H/W |
| --num-output-frames | int | 77 | Target generated frames. | VRAM up if model processes larger temporal window; runtime up always |
| --num-steps | int | 35 (or 1 in smoke mode) | Denoising/inference steps. | Peak VRAM usually near-neutral, runtime up |
| --enable-autoregressive | bool | false | Sliding-window generation for longer videos. | Peak VRAM often controlled by chunk size; runtime up |
| --chunk-size | int | 77 | Frames per autoregressive chunk. | VRAM up witsh larger chunk size |
| --chunk-overlap | int | 1 | Overlap frames between chunks. | Small VRAM effect; runtime up |

## Which Flags Increase VRAM Most

Highest impact first:
1. --resolution
2. --chunk-size (when autoregressive)
3. --num-output-frames (workload/runtime, and sometimes peak)
4. Choosing larger model/config via --model or --config-file

## Which Flags Usually Do Not Increase Peak VRAM

- --num-steps (mostly runtime)
- --seed
- --prompt, --prompt-path, --negative-prompt
- --skip-existing-output
- --keep-going
- --input-files

## Which Flags Reduce VRAM (But Shift to CPU)

- --offload-diffusion-model
- --offload-text-encoder
- --offload-tokenizer
- --offload-guardrail-models (when guardrails enabled)
- --context-parallel-size > 1 (reduces per-GPU VRAM)
