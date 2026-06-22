# Translation with Bitext Filtering

This example shows a Ray-native bitext cleaning recipe modeled after the older Canary `en-x` data cleaning workflow. It uses `DocumentBatch` and `ProcessingStage` primitives directly and keeps aligned source-target rows intact while adding `_skipme` and `reason` rejection metadata. With `--translate` it can first translate source-only rows (via the same translation stages as `run_translation_pipeline.py`) and then run them through the bitext filters.

The default input is a small `Helsinki-NLP/news_commentary` sample. You can also pass JSONL files with either `src`/`tgt` columns or Canary manifest-style `text`/`answer` columns.

## Usage

Run a local CPU smoke test:

```bash
python examples/audio/translation/run_translation_pipeline_with_filtering.py \
  --input-jsonl /path/to/bitext.jsonl \
  --output-dir /tmp/curator-bitext-cleaning
```

Run against a small News Commentary sample:

```bash
python examples/audio/translation/run_translation_pipeline_with_filtering.py \
  --dataset-config ar-cs \
  --max-rows 200 \
  --output-dir /tmp/curator-bitext-cleaning
```

Run from a Canary YAML config:

```bash
python examples/audio/translation/run_translation_pipeline_with_filtering.py \
  --config /path/to/canary_config.yaml \
  --output-dir /tmp/curator-bitext-cleaning
```

The output JSONL contains all rows plus filter metadata. Rows rejected by a stage are marked with `_skipme=1` and a `reason`; later expensive stages skip those rows.

## Optional Filters

Add M2M histogram filtering:

```bash
python examples/audio/translation/run_translation_pipeline_with_filtering.py \
  --input-jsonl /path/to/bitext.jsonl \
  --enable-histogram \
  --output-dir /tmp/curator-bitext-cleaning
```

Add FastText language ID filtering:

```bash
python examples/audio/translation/run_translation_pipeline_with_filtering.py \
  --input-jsonl /path/to/bitext.jsonl \
  --fasttext-model-path /path/to/lid.176.ftz \
  --output-dir /tmp/curator-bitext-cleaning
```

Opt into QE filtering when COMET/PyMarian dependencies and model access are available:

```bash
python examples/audio/translation/run_translation_pipeline_with_filtering.py \
  --input-jsonl /path/to/bitext.jsonl \
  --run-qe \
  --qe-models comet-qe cometoid-wmt23 \
  --output-dir /tmp/curator-bitext-cleaning
```

For PyMarian GPU runs, schedule one worker per GPU with `gpus=1.0`; this recipe does that for QE stages unless `--qe-cpu` is passed.

## Input Columns

Supported JSONL layouts:

```json
{"id":"1","src":"source sentence","tgt":"target sentence","src_lang":"en","tgt_lang":"de"}
```

```json
{"text":"source sentence","answer":"target sentence","source_lang":"en","target_lang":"de"}
```

Use `--src-field`, `--tgt-field`, `--src-lang`, and `--tgt-lang` when your column names or language metadata differ.
