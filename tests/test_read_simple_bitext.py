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

from nemo_curator.datasets.doc_dataset import ParallelDataset
from pathlib import Path
import pytest


class TestReadSimpleBitext:
    src_file = Path("tests/bitext_data/toy.de")
    tgt_file = Path("tests/bitext_data/toy.en")

    def test_pandas_read_simple_bitext(self):
        ds = ParallelDataset.read_simple_bitext(
            src_input_files=[self.src_file],
            tgt_input_files=[self.tgt_file],
            src_lang = "de",
            tgt_lang = "en",
            backend="pandas",
        )

        for idx, (src_line, tgt_line) in enumerate(zip(open(self.src_file), open(self.tgt_file))):
            assert ds.df['src'].compute()[idx] == src_line.rstrip('\n')
            assert ds.df['tgt'].compute()[idx] == tgt_line.rstrip('\n')
            assert ds.df['src_lang'].compute()[idx] == "de"
            assert ds.df['tgt_lang'].compute()[idx] == "en"

    @pytest.mark.gpu
    def test_cudf_read_simple_bitext(self):
        ds = ParallelDataset.read_simple_bitext(
            src_input_files=[self.src_file],
            tgt_input_files=[self.tgt_file],
            src_lang = "de",
            tgt_lang = "en",
            backend="cudf",
        )

        for idx, (src_line, tgt_line) in enumerate(zip(open(self.src_file), open(self.tgt_file))):
            assert ds.df['src'].compute()[idx] == src_line.rstrip('\n')
            assert ds.df['tgt'].compute()[idx] == tgt_line.rstrip('\n')
            assert ds.df['src_lang'][idx].compute()[idx] == "de"
            assert ds.df['tgt_lang'][idx].compute()[idx] == "en"
