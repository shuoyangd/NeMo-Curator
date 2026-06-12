# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Canary/news-commentary-style bitext cleaning recipe.

This recipe keeps the old Dask Canary pipeline shape while using the Ray-native
DocumentBatch and ProcessingStage APIs:

JSONL or News Commentary rows
  -> source/target word-count markers
  -> length-ratio marker
  -> optional histogram/FastText language-ID markers
  -> optional QE markers
  -> target regex cleanup
  -> JSONL writer
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import requests
from loguru import logger

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.core.client import RayClient, SlurmRayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.filters.bitext import BitextScoreFilter, LengthRatioFilter
from nemo_curator.stages.text.filters.fasttext import FastTextLangId
from nemo_curator.stages.text.filters.heuristic import WordCountFilter
from nemo_curator.stages.text.filters.histogram import HistogramFilter
from nemo_curator.stages.text.filters.qe import QualityEstimationFilter
from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter
from nemo_curator.stages.text.modifiers import DocumentModifier, Modify
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata
    from nemo_curator.stages.text.filters.doc_filter import DocumentFilter

NEWS_COMMENTARY_ROWS_URL = "https://datasets-server.huggingface.co/rows"
DEFAULT_NEWS_COMMENTARY_ROWS = 200
MAX_LOGGED_OUTPUT_PATHS = 10
QE_SCORE_FIELDS = {
    "comet-qe": "comet_qe_score",
    "cometoid-wmt23": "pymarian_qe_score",
    "cometoid-wmt23-mqm": "pymarian_mqm_qe_score",
}

REGEX_PARAMS_LIST = [
    {"pattern": "’", "repl": "'"},
    {"pattern": "‘", "repl": "'"},
    {"pattern": "—", "repl": "-"},
    {"pattern": "–", "repl": "-"},
    {"pattern": "-", "repl": "-"},
    {"pattern": "_", "repl": " "},
    {"pattern": "——", "repl": "-"},
    {"pattern": "Ё", "repl": "Е"},
    {"pattern": "ё", "repl": "е"},
    {"pattern": "♫", "repl": " "},
    {"pattern": "♪", "repl": " "},
    {"pattern": "♬", "repl": " "},
    {"pattern": "♩", "repl": " "},
    {"pattern": "♭", "repl": " "},
    {"pattern": r"\|", "repl": " "},
    {"pattern": ";", "repl": ","},
    {"pattern": r"\[[^\]]*\]", "repl": ""},
    {"pattern": r" ?\([^\)]+\)", "repl": ""},
    {"pattern": r" ?{[^}]+}", "repl": ""},
    {
        "pattern": "[^ !$%',-.0123456789;?ABCDEFGHIJKLMNOPQRSßTUVWXYŸZabcdefghijklmnopqrsẞtuvwxyÿz¡£¿ÀÁÂÃÄÅÆÇÈÉÊÌÍÎÑÒÓÔÕÖØÙÚÜÝàáâãäåæçèéêëìíîïñòóôõöøùúûüýĀāĂăĄąĆćĊċČčĎďĐđĒēĖėĘęĚěĠġĢģĦħĪīĮįĶķĹĺĻļĽľŁłŃńŅņŇňŐőŒœŔŕŘřŚśŠšŤťŪūŮůŰűŲųŹźŻżŽžȘșȚțΆΈΉΌΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩάέήίαβγδεζηθικλμνξοπρστυφχψωϊόύώЁЄІЇАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяёєіїҐґ€₴₽/:]",
        "repl": "",
    },
    {"pattern": r"\s+\.", "repl": "."},
    {"pattern": r"\?+", "repl": "?"},
    {"pattern": r"\.+", "repl": "."},
    {"pattern": ",+", "repl": ","},
    {"pattern": "!+", "repl": "!"},
    {"pattern": r"\s+", "repl": " "},
    {"pattern": r" ([.,!?])", "repl": r"\1"},
    {"pattern": " '", "repl": "'"},
    {"pattern": "' ", "repl": "'"},
    {"pattern": "'+", "repl": "'"},
    {"pattern": " - ", "repl": "-"},
]


@dataclass
class FieldScoreMarker(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Apply a DocumentFilter to one field while preserving marked rows."""

    filter_obj: DocumentFilter
    text_field: str
    score_field: str
    reason: str
    skip_field: str = "_skipme"
    reason_field: str = "reason"
    mark_only: bool = True
    name: str = "field_score_marker"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.text_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        fields = [self.score_field]
        if self.mark_only:
            fields.extend([self.skip_field, self.reason_field])
        return ["data"], fields

    def ray_stage_spec(self) -> dict[str, Any]:
        requires_setup = hasattr(self.filter_obj, "load_model") or hasattr(self.filter_obj, "load_tokenizer")
        return {"is_actor_stage": requires_setup}

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        if hasattr(self.filter_obj, "model_check_or_download"):
            self.filter_obj.model_check_or_download()

    def setup(self, _: WorkerMetadata | None = None) -> None:
        if hasattr(self.filter_obj, "load_model"):
            self.filter_obj.load_model()
        if hasattr(self.filter_obj, "load_tokenizer"):
            self.filter_obj.load_tokenizer()

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas().copy()
        if df.empty:
            return self._make_output_batch(batch, df)

        process_mask = self._get_process_mask(df)
        scores = df.loc[process_mask, self.text_field].astype(str).apply(self.filter_obj.score_document)
        if self.score_field not in df.columns:
            df[self.score_field] = pd.NA
        df.loc[scores.index, self.score_field] = scores

        keep_mask = scores.apply(self.filter_obj.keep_document)
        if self.mark_only:
            processed_indices = keep_mask.index
            rejected_indices = keep_mask[~keep_mask].index
            df.loc[processed_indices, self.skip_field] = 0
            df.loc[rejected_indices, self.skip_field] = 1
            df.loc[rejected_indices, self.reason_field] = self.reason
        else:
            df = df.loc[keep_mask[keep_mask].index]

        return self._make_output_batch(batch, df)

    def _get_process_mask(self, df: pd.DataFrame) -> pd.Series:
        if not self.mark_only:
            return pd.Series(True, index=df.index)

        if self.skip_field not in df.columns:
            df[self.skip_field] = 0
        if self.reason_field not in df.columns:
            df[self.reason_field] = None
        return df[self.skip_field].fillna(0).eq(0)

    def _make_output_batch(self, batch: DocumentBatch, df: pd.DataFrame) -> DocumentBatch:
        return DocumentBatch(
            task_id=f"{batch.task_id}_{self.name}",
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


class RegexSubstitutionModifier(DocumentModifier):
    """Apply a fixed sequence of regex substitutions to one text field."""

    def __init__(self, regex_params_list: list[dict[str, str]]) -> None:
        super().__init__()
        self._name = "regex_substitution"
        self._substitutions = [
            (re.compile(params["pattern"]), params["repl"]) for params in regex_params_list
        ]

    def modify_document(self, text: object) -> str:
        value = "" if text is None else str(text)
        for pattern, repl in self._substitutions:
            value = pattern.sub(repl, value)
        return value.strip()


def expand_file_list(filename: str) -> list[str]:
    pattern = re.compile(r"(.*)_OP_(\d+)..(\d+)_CL_(.*)")
    match = pattern.match(filename)

    if not match:
        return [filename]

    prefix = match.group(1)
    start = int(match.group(2))
    end = int(match.group(3))
    suffix = match.group(4)
    return [f"{prefix}{idx}{suffix}" for idx in range(start, end + 1)]


def manifest_paths_from_config(config_path: str) -> list[str]:
    try:
        import yaml
    except ImportError as e:
        msg = "Reading Canary YAML configs requires PyYAML"
        raise ImportError(msg) from e

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, list) or not config:
        msg = "Expected a Canary config list with at least one section"
        raise ValueError(msg)

    translation_config = config[0] if len(config) == 1 else config[1]
    input_cfg = translation_config.get("input_cfg", [])
    paths: list[str] = []
    for item in input_cfg:
        if "manifest_filepath" not in item:
            msg = f"Missing manifest_filepath in input_cfg entry: {item}"
            raise ValueError(msg)
        paths.extend(expand_file_list(str(item["manifest_filepath"])))

    if not paths:
        msg = f"No manifest paths found in {config_path}"
        raise ValueError(msg)
    return paths


def resolve_text_field(
    df: pd.DataFrame,
    explicit_field: str | None,
    candidates: tuple[str, ...],
    label: str,
    required: bool = True,
) -> str | None:
    if explicit_field is not None:
        if explicit_field not in df.columns:
            msg = f"Input is missing explicit {label} field '{explicit_field}'"
            raise ValueError(msg)
        return explicit_field

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    if not required:
        return None

    msg = f"Input must contain one of {candidates} or pass --{label.replace('_', '-')}-field"
    raise ValueError(msg)


def resolve_language_values(
    df: pd.DataFrame,
    explicit_lang: str | None,
    candidates: tuple[str, ...],
    label: str,
) -> pd.Series:
    if explicit_lang is not None:
        return pd.Series([explicit_lang.lower()] * len(df), index=df.index)

    for candidate in candidates:
        if candidate in df.columns:
            return df[candidate].fillna("").astype(str).str[:2].str.lower()

    msg = f"Could not infer {label}; pass --{label.replace('_', '-')} or include one of {candidates}"
    raise ValueError(msg)


def normalize_bitext_frame(df: pd.DataFrame, args: argparse.Namespace, source_name: str) -> pd.DataFrame:
    src_field = resolve_text_field(df, args.src_field, ("src", "text"), "src")
    tgt_field = resolve_text_field(df, args.tgt_field, ("tgt", "answer"), "tgt")

    source_file = str(source_name)
    ids = (
        df["id"].astype(str)
        if "id" in df.columns
        else pd.Series([f"{Path(source_name).stem}:{idx}" for idx in range(len(df))], index=df.index)
    )
    records = pd.DataFrame(
        {
            "id": ids,
            "src": df[src_field].fillna("").astype(str),
            "tgt": df[tgt_field].fillna("").astype(str),
            "src_lang": resolve_language_values(df, args.src_lang, ("src_lang", "source_lang"), "src_lang"),
            "tgt_lang": resolve_language_values(df, args.tgt_lang, ("tgt_lang", "target_lang"), "tgt_lang"),
            "source_file": source_file,
        }
    )
    return records


def read_jsonl_records(paths: list[str], args: argparse.Namespace) -> list[dict[str, Any]]:
    frames = []
    for path in paths:
        df = pd.read_json(path, lines=True)
        frames.append(normalize_bitext_frame(df, args, path))

    if not frames:
        return []

    return pd.concat(frames, ignore_index=True).to_dict(orient="records")


def news_commentary_langs(args: argparse.Namespace) -> tuple[str, str]:
    config_langs = args.dataset_config.split("-")
    if len(config_langs) != 2:
        msg = "--dataset-config must be a two-language News Commentary config such as ar-cs"
        raise ValueError(msg)
    return args.src_lang or config_langs[0], args.tgt_lang or config_langs[1]


def fetch_news_commentary_records(args: argparse.Namespace) -> list[dict[str, str]]:
    src_lang, tgt_lang = news_commentary_langs(args)
    max_rows = args.max_rows if args.max_rows is not None else DEFAULT_NEWS_COMMENTARY_ROWS
    records: list[dict[str, str]] = []
    offset = 0

    while len(records) < max_rows:
        length = min(args.page_size, max_rows - len(records))
        response = requests.get(
            NEWS_COMMENTARY_ROWS_URL,
            params={
                "dataset": "Helsinki-NLP/news_commentary",
                "config": args.dataset_config,
                "split": "train",
                "offset": offset,
                "length": length,
            },
            timeout=60,
        )
        response.raise_for_status()
        rows = response.json()["rows"]
        if not rows:
            break

        for row in rows:
            translation = row["row"]["translation"]
            records.append(
                {
                    "id": str(row["row"].get("id", offset)),
                    "src": translation[src_lang],
                    "tgt": translation[tgt_lang],
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang,
                    "source_file": f"Helsinki-NLP/news_commentary:{args.dataset_config}",
                }
            )
        offset += len(rows)

    return records


def repeat_records(records: list[dict[str, Any]], repeat: int) -> list[dict[str, Any]]:
    if repeat <= 1:
        return records

    repeated = []
    for repeat_idx in range(repeat):
        for record in records:
            repeated_record = dict(record)
            repeated_record["id"] = f"{record['id']}:r{repeat_idx}"
            repeated.append(repeated_record)
    return repeated


def limit_records(records: list[dict[str, Any]], max_rows: int | None) -> list[dict[str, Any]]:
    if max_rows is None:
        return records
    return records[:max_rows]


def make_document_batches(records: list[dict[str, Any]], rows_per_task: int, dataset_name: str) -> list[DocumentBatch]:
    tasks = []
    for start_idx in range(0, len(records), rows_per_task):
        shard = records[start_idx : start_idx + rows_per_task]
        shard_id = start_idx // rows_per_task
        source_files = sorted({str(record.get("source_file", dataset_name)) for record in shard})
        tasks.append(
            DocumentBatch(
                task_id=f"{dataset_name}_{shard_id:06d}",
                dataset_name=dataset_name,
                data=pd.DataFrame(shard),
                _metadata={"source_files": source_files},
            )
        )
    return tasks


def build_initial_tasks(args: argparse.Namespace) -> tuple[list[DocumentBatch], str, str]:
    if args.config:
        paths = manifest_paths_from_config(args.config)
        records = read_jsonl_records(paths, args)
    elif args.input_jsonl:
        records = read_jsonl_records(args.input_jsonl, args)
    else:
        records = fetch_news_commentary_records(args)

    records = limit_records(records, args.max_rows if args.input_jsonl or args.config else None)
    records = repeat_records(records, args.repeat)
    if not records:
        msg = "No input records were loaded"
        raise ValueError(msg)

    src_lang = args.src_lang or str(records[0]["src_lang"])
    tgt_lang = args.tgt_lang or str(records[0]["tgt_lang"])
    tasks = make_document_batches(records, args.rows_per_task, args.dataset_name)
    logger.info(f"Prepared {len(records)} rows in {len(tasks)} DocumentBatch task(s)")
    logger.info(f"Using language pair {src_lang}-{tgt_lang}")
    return tasks, src_lang, tgt_lang


def qe_cutoff_for_model(model_name: str, args: argparse.Namespace) -> float:
    if model_name == "comet-qe":
        return args.comet_cutoff
    if model_name.startswith("cometoid"):
        return args.pymarian_cutoff
    msg = f"Unsupported QE model: {model_name}"
    raise ValueError(msg)


def add_field_marker(
    pipeline: Pipeline,
    filter_obj: DocumentFilter,
    text_field: str,
    score_field: str,
    reason: str,
    cpus: float,
) -> None:
    pipeline.add_stage(
        FieldScoreMarker(
            filter_obj=filter_obj,
            text_field=text_field,
            score_field=score_field,
            reason=reason,
            name=f"{score_field}_marker",
        ).with_(resources=Resources(cpus=cpus))
    )


def add_optional_field_filters(
    pipeline: Pipeline,
    args: argparse.Namespace,
    src_lang: str,
    tgt_lang: str,
) -> list[str]:
    fields = []

    if args.enable_histogram:
        cache_dir = args.histogram_cache_dir or str(Path(args.output_dir) / "histogram_cache")
        add_field_marker(
            pipeline,
            HistogramFilter(lang=src_lang, threshold=args.histogram_threshold, cache_dir=cache_dir),
            "src",
            "src_histogram_score",
            "src_HistogramFilter",
            args.cpu_stage_cpus,
        )
        add_field_marker(
            pipeline,
            HistogramFilter(lang=tgt_lang, threshold=args.histogram_threshold, cache_dir=cache_dir),
            "tgt",
            "tgt_histogram_score",
            "tgt_HistogramFilter",
            args.cpu_stage_cpus,
        )
        fields.extend(["src_histogram_score", "tgt_histogram_score"])

    if args.fasttext_model_path:
        add_field_marker(
            pipeline,
            FastTextLangId(
                model_path=args.fasttext_model_path,
                min_langid_score=args.fasttext_min_score,
                expected_lang=src_lang,
            ),
            "src",
            "src_fasttext_langid",
            "src_FastTextLangId",
            args.cpu_stage_cpus,
        )
        add_field_marker(
            pipeline,
            FastTextLangId(
                model_path=args.fasttext_model_path,
                min_langid_score=args.fasttext_min_score,
                expected_lang=tgt_lang,
            ),
            "tgt",
            "tgt_fasttext_langid",
            "tgt_FastTextLangId",
            args.cpu_stage_cpus,
        )
        fields.extend(["src_fasttext_langid", "tgt_fasttext_langid"])

    return fields


def add_qe_filters(pipeline: Pipeline, args: argparse.Namespace, output_fields: list[str]) -> None:
    if not args.run_qe:
        return

    for model_name in args.qe_models:
        score_field = QE_SCORE_FIELDS[model_name]
        output_fields.append(score_field)
        model_kwargs: dict[str, Any] = {}
        if model_name.startswith("cometoid"):
            model_kwargs["shard_size"] = args.pymarian_shard_size
            if args.pymarian_args:
                model_kwargs["marian_args"] = args.pymarian_args

        qe_stage = QualityEstimationFilter(
            model_name=model_name,
            cutoff=qe_cutoff_for_model(model_name, args),
            mode=args.qe_mode,
            score_field=score_field,
            mark_only=True,
            gpu=not args.qe_cpu,
            model_kwargs=model_kwargs,
        )
        if not args.qe_cpu:
            qe_stage = qe_stage.with_(
                name=f"{model_name.replace('-', '_')}_filter",
                resources=Resources(cpus=args.qe_cpus, gpus=1.0),
                batch_size=args.qe_batch_size,
            )
        pipeline.add_stage(qe_stage)


def build_pipeline(args: argparse.Namespace, src_lang: str, tgt_lang: str) -> Pipeline:
    pipeline = Pipeline(
        name="bitext_cleaning_recipe",
        description="Canary/news-commentary-style bitext cleaning with Ray DocumentBatch stages",
    )

    output_fields = [
        "id",
        "src",
        "tgt",
        "src_lang",
        "tgt_lang",
        "source_file",
        "src_word_count",
        "tgt_word_count",
        "length_ratio",
        "_skipme",
        "reason",
    ]

    add_field_marker(
        pipeline,
        WordCountFilter(min_words=args.min_words, lang=src_lang),
        "src",
        "src_word_count",
        "src_WordCountFilter",
        args.cpu_stage_cpus,
    )
    add_field_marker(
        pipeline,
        WordCountFilter(min_words=args.min_words, lang=tgt_lang),
        "tgt",
        "tgt_word_count",
        "tgt_WordCountFilter",
        args.cpu_stage_cpus,
    )
    pipeline.add_stage(
        BitextScoreFilter(
            LengthRatioFilter(max_ratio=args.length_max_ratio, src_lang=src_lang, tgt_lang=tgt_lang),
            score_field="length_ratio",
            mark_only=True,
        ).with_(resources=Resources(cpus=args.cpu_stage_cpus))
    )

    output_fields.extend(add_optional_field_filters(pipeline, args, src_lang, tgt_lang))
    add_qe_filters(pipeline, args, output_fields)

    if not args.skip_regex_cleanup:
        pipeline.add_stage(
            Modify(
                RegexSubstitutionModifier(REGEX_PARAMS_LIST),
                input_fields=args.cleanup_field,
                output_fields=args.cleanup_field,
            ).with_(resources=Resources(cpus=args.cpu_stage_cpus))
        )

    pipeline.add_stage(JsonlWriter(path=args.output_dir, fields=output_fields, mode=args.output_mode))
    return pipeline


def summarize_results(results: list[Any] | None, start_time: float) -> None:
    elapsed = time.time() - start_time
    if not results:
        logger.warning("Pipeline returned no output tasks")
        return

    output_paths = [path for task in results for path in task.data]
    logger.info(f"Finished bitext cleaning in {elapsed:.2f}s")
    logger.info(f"Wrote {len(output_paths)} output shard(s)")
    for path in output_paths[:MAX_LOGGED_OUTPUT_PATHS]:
        logger.info(f"  {path}")
    if len(output_paths) > MAX_LOGGED_OUTPUT_PATHS:
        logger.info(f"  ... {len(output_paths) - MAX_LOGGED_OUTPUT_PATHS} more")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ray bitext cleaning recipe")
    parser.add_argument("--slurm", action="store_true", help="Use SlurmRayClient; pass this from srun")
    parser.add_argument("--config", help="Optional Canary YAML config with manifest_filepath entries")
    parser.add_argument("--input-jsonl", nargs="+", help="JSONL inputs with src/tgt or text/answer columns")
    parser.add_argument("--src-field", help="Source text field for JSONL inputs")
    parser.add_argument("--tgt-field", help="Target text field for JSONL inputs")
    parser.add_argument("--dataset-config", default="ar-cs", help="News Commentary config used when no input is given")
    parser.add_argument("--src-lang", help="Override or supply source language code")
    parser.add_argument("--tgt-lang", help="Override or supply target language code")
    parser.add_argument("--dataset-name", default="bitext_cleaning")
    parser.add_argument("--max-rows", type=int, help="Maximum rows to load; News Commentary defaults to 200")
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--rows-per-task", type=int, default=128)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-mode", choices=["ignore", "overwrite", "append", "error"], default="overwrite")
    parser.add_argument("--min-words", type=int, default=4)
    parser.add_argument("--length-max-ratio", type=float, default=9.0)
    parser.add_argument("--cpu-stage-cpus", type=float, default=1.0)
    parser.add_argument("--enable-histogram", action="store_true")
    parser.add_argument("--histogram-threshold", type=float, default=0.8)
    parser.add_argument("--histogram-cache-dir")
    parser.add_argument("--fasttext-model-path")
    parser.add_argument("--fasttext-min-score", type=float, default=0.5)
    parser.add_argument("--run-qe", action="store_true", help="Opt into COMET/PyMarian QE filtering")
    parser.add_argument("--qe-models", nargs="+", default=["cometoid-wmt23"], choices=sorted(QE_SCORE_FIELDS))
    parser.add_argument("--qe-mode", choices=["simple", "always_en_x", "bidi"], default="simple")
    parser.add_argument("--qe-cpu", action="store_true")
    parser.add_argument("--qe-cpus", type=float, default=1.0)
    parser.add_argument("--qe-batch-size", type=int, default=1)
    parser.add_argument("--comet-cutoff", type=float, default=-0.5)
    parser.add_argument("--pymarian-cutoff", type=float, default=0.6)
    parser.add_argument("--pymarian-shard-size", type=int, default=5000)
    parser.add_argument("--pymarian-args")
    parser.add_argument("--cleanup-field", default="tgt", choices=["src", "tgt"])
    parser.add_argument("--skip-regex-cleanup", action="store_true")
    parser.add_argument("--worker-connect-timeout-s", type=int, default=900)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config and args.input_jsonl:
        msg = "Pass only one of --config or --input-jsonl"
        raise ValueError(msg)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    ray_client = SlurmRayClient(worker_connect_timeout_s=args.worker_connect_timeout_s) if args.slurm else RayClient()
    ray_client.start()

    start_time = time.time()
    try:
        initial_tasks, src_lang, tgt_lang = build_initial_tasks(args)
        pipeline = build_pipeline(args, src_lang, tgt_lang)
        logger.info(f"\n{pipeline.describe()}")
        results = pipeline.run(executor=RayDataExecutor(), initial_tasks=initial_tasks)
    finally:
        ray_client.stop()

    summarize_results(results, start_time)


if __name__ == "__main__":
    main()
