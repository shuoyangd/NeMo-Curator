# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import json
import os
import re
import shutil
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, List, Optional, Tuple

import yaml

from nemo_curator import ParallelScoreFilter, Sequential
from nemo_curator.datasets import ParallelDataset
from nemo_curator.filters import (
    FastTextLangId,
    HistogramFilter,
    LengthRatioFilter,
    QualityEstimationFilter,
    WordCountFilter,
)
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.script_utils import ArgumentHelper

SCRIPT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR_PATH, "data")
TEMP_DIR = ""
FAST_TEXT_MODEL_DIR = ""


def peek_manifest_for_language(filename: str) -> Tuple[str]:
    with open(filename) as fin:
        dp_peek = json.loads(fin.readline())
        return dp_peek["source_lang"], dp_peek["target_lang"]


def expand_file_list(filename: str) -> List[str]:
    pattern = re.compile(r"(.*)_OP_(\d+)..(\d+)_CL_(.*)")
    match = pattern.match(filename)

    if match:
        prefix = match.group(1)  # Prefix
        start = int(match.group(2))  # Start of the range
        end = int(match.group(3))  # End of the range
        suffix = match.group(4)  # Suffix

        return [f"{prefix}{i}{suffix}" for i in range(start, end + 1)]
    else:
        return [filename]


def filter_dataset(
    dataset: ParallelDataset, src_lang: str, tgt_lang: str, gpu: bool = False
) -> ParallelDataset:
    filters = Sequential(
        [
            ParallelScoreFilter(
                WordCountFilter(min_words=1, lang=src_lang),  # filter out empty lines
                WordCountFilter(min_words=1, lang=tgt_lang),  # filter out empty lines
                src_field="text",
                tgt_field="answer",
                score_type=int,
                add_skip_label_only=True,
            ),
            LengthRatioFilter(
                max_ratio=4,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                score_field="length_ratio",
                score_type=float,
                src_field="text",
                tgt_field="answer",
                add_skip_label_only=True,
            ),
            ParallelScoreFilter(
                HistogramFilter(lang=src_lang),
                HistogramFilter(lang=tgt_lang),
                src_score="src_hist",
                tgt_score="tgt_hist",
                src_field="text",
                tgt_field="answer",
                score_type=int,
                add_skip_label_only=True,
            ),
            ParallelScoreFilter(
                FastTextLangId(model_path=FAST_TEXT_MODEL_DIR, lang=src_lang),
                FastTextLangId(model_path=FAST_TEXT_MODEL_DIR, lang=tgt_lang),
                src_field="text",
                tgt_field="answer",
                score_type=str,
                add_skip_label_only=True,
            ),
            QualityEstimationFilter(
                "cometoid-wmt23",
                cutoff=0.75,
                gpu=gpu,
                mode="simple",
                src_field="text",
                tgt_field="answer",
                metadata_fields=["source_lang", "target_lang"],
                metadata_field_name_mapping={
                    "source_lang": "src_lang",
                    "target_lang": "tgt_lang",
                },
                score_type=float,
                add_skip_label_only=True,
            ),
        ]
    )
    filtered_dataset = filters(dataset)
    return filtered_dataset


def run_curation_pipeline(
    args: Any, files: List[str], group_idx: Optional[int] = None
) -> None:
    # Initialize the Dask cluster.
    client = get_client(**ArgumentHelper.parse_client_args(args))

    print("Reading the data...")

    bitext_dataset = ParallelDataset.read_json(
        files,
        add_filename=True,
    )
    src_lang, tgt_lang = peek_manifest_for_language(files[0])

    print("Executing the pipeline...")
    curation_steps = Sequential(
        [
            partial(
                filter_dataset,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                gpu=(args.device == "gpu"),
            ),
        ]
    )

    dataset = curation_steps(bitext_dataset)
    dataset = dataset.persist()

    print(f"Original dataset length: {len(bitext_dataset.df)}")
    print(f"After dataprep: {len(dataset.df)}")
    print("Writing the results to disk...")

    # raise NotImplementedError("writing not finished yet, filters not checked")

    # Overwrite existing files in the curated directory.
    if group_idx is not None:
        out_path = os.path.join(args.output, f"group{group_idx}")
    else:
        out_path = args.output

    if os.path.isdir(out_path):
        shutil.rmtree(out_path)

    os.makedirs(out_path)
    dataset.to_json(out_path, write_to_filename=True)
    client.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="yaml config for canary training",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="output folder for curated data",
    )
    args = ArgumentHelper(parser).add_distributed_args().parse_args()
    # Limit the total number of workers to ensure we don't run out of memory.
    args.n_workers = min(args.n_workers, 8)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # assuming config[1] is the translation section
    # if it doesn't exist, translate using the input from the asr section
    if len(config) == 1:
        translation_input_config = config[0]
    elif len(config) >= 1:
        translation_input_config = config[1]
    else:
        raise RuntimeError(
            "I got what appears to be an empty yaml config. Can you check the format?"
        )

    unexpanded_file_paths = [
        cfg["manifest_filepath"] for cfg in translation_input_config["input_cfg"]
    ]
    for group_idx, unexpanded_file_path in enumerate(unexpanded_file_paths):
        file_paths = []
        with TemporaryDirectory(dir=TEMP_DIR) as tempdir:
            for path in expand_file_list(unexpanded_file_path):
                path = Path(path)
                file_basename = path.stem + ".jsonl"
                symlink = os.path.join(tempdir, file_basename)
                os.symlink(path, symlink)
                file_paths.append(symlink)

            run_curation_pipeline(args, file_paths, group_idx)

            with open(
                os.path.join(args.output, f"group{group_idx}", "ORIG_PATH"), "w"
            ) as f:
                f.write(os.path.dirname(unexpanded_file_path) + os.linesep)


if __name__ == "__main__":
    main()
