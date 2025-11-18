# Evaluation Toolkit

Tools for generating evaluation corpora from LLaDA checkpoints and scoring them
with a reference LM (judge) to compute cross-model perplexity.

| File | Purpose |
| --- | --- |
| `pipeline.py` | Use `generation.py`'s diffusion sampler plus the judge scorer to produce the full report. |
| `perplexity.py` | Score any JSONL/dataset using an AutoModel judge to obtain perplexity stats. |
| `speedtest.py` | Measure raw diffusion generation throughput (tokens/s) for a checkpoint. |
| `speedtest_ar.py` | Benchmark autoregressive HF models with/without KV cache. |

## Quick start

```bash
PYTHONPATH=$(pwd) python eval/pipeline.py \
  --student-checkpoint output/tllada_50m_dl/base/checkpoint-331210 \
  --judge-model Qwen/Qwen3-1.7B \
  --output-dir eval/runs/llada_50m_base_ppl \
  --generation-num-prompts 256 \
  --generation-max-new-tokens 128 \
  --generation-diffusion-steps 128 \
  --generation-block-size 128 \
  --judge-batch-size 2 \
  --judge-max-context 512 \
  --reference-split train[:2000] \
  --reference-max-samples 100
```

> The `PYTHONPATH=$(pwd)` ensures `generation.py` and other repo modules are importable.
> By default the student model loads in FP16 for generation to keep memory low; override via `--generation-dtype`.

All outputs land under a run-specific subdirectory named
`gen_{max_new}_{steps}_{block}`. The example above writes to
`eval/runs/llada_40m_dl_ppl/gen_128_32_8/`.

### Outputs

Running the pipeline produces:

- `artifacts/model_generations.jsonl`: prompt text/ids, generated ids, decoded outputs.
- `artifacts/model_generations_perplexity.{json,jsonl}`: summary + per-sample judge perplexity on generations.
- `artifacts/reference_perplexity.{json,jsonl}`: summary + per-sample judge perplexity on TinyStories slices.
- `perplexity_summary.json`: top-level metadata pointing to all artifacts.

### Standalone scoring

If you already have generations (from the pipeline or elsewhere), run:

```bash
python eval/perplexity.py \
  --judge-model Qwen/Qwen3-1.7B \
  --input-jsonl path/to/generations.jsonl \
  --text-field generated_text \
  --output-json path/to/ppl.json
```
### Speed test (diffusion)

```bash
PYTHONPATH=$(pwd) python eval/speedtest.py \
  --model-path output/llada_40m_dl/checkpoint-463694 \
  --num-prompts 256 \
  --batch-size 32 \
  --max-new-tokens 128 \
  --diffusion-steps 128 \
  --block-size 32
```

### Speed test (autoregressive)

```bash
PYTHONPATH=$(pwd) python3 eval/speedtest_ar.py \
  --num-prompts 256 \
  --batch-size 32 \
  --prompt-max-tokens 512 \
  --max-new-tokens 128 \
  --dtype fp16 \
  --device cuda \
  --no-kv-cache
```

This benchmarks `rzzhan/tiny-llama-stories-42m` by default using TinyStories prompts; override `--model-name` or `--dataset-name` as needed.
