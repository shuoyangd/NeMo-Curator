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
import pandas as pd
from dask.dataframe.extensions import make_array_nonempty
from dask.typing import no_default
from dask.array import logical_and

from nemo_curator.datasets import DocumentDataset, ParallelDataset
from nemo_curator.utils.module_utils import is_batched

# Override so that pd.NA is not passed during the metadata inference
make_array_nonempty.register(
    pd.StringDtype,
    lambda x: pd.array(["a", "b"], dtype=x),
)


class Score:
    def __init__(self, score_fn, score_field, text_field="text", score_type=None):
        """
        Args:
          score_fn: The score function that takes in a document string and outputs a score for the document
          score_field: The field the score will be stored in.
          text_field: The field the documents will be read from
        """
        self.score_fn = score_fn
        self.score_field = score_field
        self.text_field = text_field
        self.score_type = score_type

    def __call__(self, dataset):
        # Set the metadata for the function calls if provided
        if self.score_type:
            meta = (None, self.score_type)
        else:
            meta = no_default

        if is_batched(self.score_fn):
            dataset.df[self.score_field] = dataset.df[self.text_field].map_partitions(
                self.score_fn, meta=meta
            )
        else:
            dataset.df[self.score_field] = dataset.df[self.text_field].apply(
                self.score_fn, meta=meta
            )

        return dataset


class Filter:
    def __init__(self, filter_fn, filter_field, invert=False):
        """
        Args:
          filter_fn: A function that returns True if the document is to be kept
          filter_field: The field(s) to be passed into the filter function.
          invert: Whether to invert the filter condition
        """
        self.filter_fn = filter_fn
        self.filter_field = filter_field
        self.invert = invert

    def __call__(self, dataset):
        if is_batched(self.filter_fn):
            bool_mask = dataset.df[self.filter_field].map_partitions(
                self.filter_fn, meta=(None, bool)
            )
        else:
            bool_mask = dataset.df[self.filter_field].apply(
                self.filter_fn, meta=(None, bool)
            )

        if self.invert:
            bool_mask = ~bool_mask

        return DocumentDataset(dataset.df[bool_mask])


class ScoreFilter:
    def __init__(
        self,
        filter_obj,
        text_field="text",
        score_field=None,
        score_type=None,
        invert=False,
    ):
        """
        Args:
          score_field: The field to which the scores will be written. If None, scores will be immediately discarded after use.
        """
        self.filter_obj = filter_obj
        self.text_field = text_field
        self.score_field = score_field
        self.score_type = score_type
        self.invert = invert

    def __call__(self, dataset):
        # Set the metadata for the function calls if provided
        if self.score_type:
            meta = (None, self.score_type)
        else:
            meta = no_default

        if is_batched(self.filter_obj.score_document):
            scores = dataset.df[self.text_field].map_partitions(
                self.filter_obj.score_document, meta=meta
            )
        else:
            scores = dataset.df[self.text_field].apply(
                self.filter_obj.score_document, meta=meta
            )

        if self.score_field is not None:
            dataset.df[self.score_field] = scores

        if is_batched(self.filter_obj.keep_document):
            bool_mask = scores.map_partitions(
                self.filter_obj.keep_document, meta=(None, bool)
            )
        else:
            bool_mask = scores.apply(self.filter_obj.keep_document, meta=(None, bool))
        if self.invert:
            bool_mask = ~bool_mask

        return DocumentDataset(dataset.df[bool_mask])


class JointScoreFilter:
    def __init__(
        self,
        filter_obj,
        src_field="src",
        tgt_field="tgt",
        score_field=None,
        score_type=None,
        invert=False,
    ):
        """
        A filter object wrapper class for applying bilingual filter objects (such as length ratio, QE filter) on bitext.

        Args:
          filter_obj: Needs to be a filter that applies to 2 text columns, as is the case in bitext.
          score_field: The field to which the scores will be written. If None, scores will be immediately discarded after use.
        """
        self.filter_obj = filter_obj
        self.src_field = src_field
        self.tgt_field = tgt_field
        self.score_field = score_field
        self.score_type = score_type
        self.invert = invert
        
    def __call__(self, dataset: ParallelDataset):
        # Set the metadata for the function calls if provided
        if self.score_type:
            meta = (None, self.score_type)
        else:
            meta = no_default
        if is_batched(self.filter_obj.score_document):
            scores = dataset.df[[self.src_field, self.tgt_field]].map_partitions(
                self.filter_obj.score_document, meta=meta
            )
        else:
            scores = dataset.df[[self.src_field, self.tgt_field]].apply(
                self.filter_obj.score_document, axis=1, meta=meta
            )

        if self.score_field is not None:
            dataset.df[self.score_field] = scores
            
        if is_batched(self.filter_obj.keep_document):
            bool_mask = scores.map_partitions(
                self.filter_obj.keep_document, meta=(None, bool)
            )
        else:
            bool_mask = scores.apply(self.filter_obj.keep_document, meta=(None, bool))
        if self.invert:
            bool_mask = ~bool_mask

        return ParallelDataset(dataset.df[bool_mask])


class ParallelScoreFilter:
    def __init__(
        self,
        src_filter_obj,
        tgt_filter_obj,
        src_field="src",
        tgt_field="tgt",
        src_score=None,
        tgt_score=None,
        score_type=None,
        invert=False,
    ):
        """
        A filter object wrapper class for applying monolingual filter objects on bitext.
        If either side of the bitext is discarded, the whole bitext pair is discarded.

        Args:
          score_field: The field to which the scores will be written. If None, scores will be immediately discarded after use.
        """
        self.src_filter_obj = src_filter_obj
        self.tgt_filter_obj = tgt_filter_obj
        self.src_field = src_field
        self.tgt_field = tgt_field
        self.src_score = src_score
        self.tgt_score = tgt_score
        self.score_type = score_type
        self.invert = invert

    def __call__(self, dataset: ParallelDataset):
        # Set the metadata for the function calls if provided
        if self.score_type:
            meta = (None, self.score_type)
        else:
            meta = no_default

        scores_list = {}
        for filter_obj, field in [
                (self.src_filter_obj, self.src_field),
                (self.tgt_filter_obj, self.tgt_field)
        ]:
            if is_batched(filter_obj.score_document):
                scores = dataset.df[field].map_partitions(
                    filter_obj.score_document, meta=meta
                )
                scores_list[field] = scores
            else:
                scores = dataset.df[field].apply(
                    filter_obj.score_document, meta=meta
                )
                scores_list[field] = scores

        if self.src_score is not None:
            dataset.df[self.src_score] = scores_list[self.src_field]
        if self.tgt_score is not None:
            dataset.df[self.tgt_score] = scores_list[self.tgt_field]

        bool_masks = {}
        for filter_obj, field in [
                (self.src_filter_obj, self.src_field),
                (self.tgt_filter_obj, self.tgt_field)
        ]:
            if is_batched(filter_obj.keep_document):
                bool_mask = scores_list[field].map_partitions(
                    filter_obj.keep_document, meta=(None, bool)
                )
                bool_masks[field] = bool_mask
            else:
                bool_mask = scores_list[field].apply(
                    filter_obj.keep_document, meta=(None, bool)
                )
                bool_masks[field] = bool_mask
        if self.invert:
            for field in [self.src_field, self.tgt_field]:
                bool_masks[field] = ~bool_masks[field]
            bool_mask = ~bool_mask

        # remove lines together if one of them is filtered
        bool_mask_joint = logical_and(bool_masks[self.src_field], bool_masks[self.tgt_field])
        
        return ParallelDataset(dataset.df[bool_mask_joint])
