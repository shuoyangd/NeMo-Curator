# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import re
from typing import Dict, List

from nemo_curator.modifiers import DocumentModifier

__all__ = ["RegexModifier"]


class RegexModifier(DocumentModifier):

    def __init__(self, regex_params_list: List[Dict]):
        super().__init__()
        self.regex_params_list = regex_params_list

        # verify all dicts in regex_params_list have "pattern" and "repl" keys
        for regex_params_dict in self.regex_params_list:
            if not "pattern" in regex_params_dict.keys():
                raise ValueError(
                    f"Need to have key 'pattern' in all entries of `regex_params_list`: {self.regex_params_list}"
                )
            if not "repl" in regex_params_dict.keys():
                raise ValueError(
                    f"Need to have key 'repl' in all entries of `regex_params_list`: {self.regex_params_list}"
                )

    # def process_dataset_entry(self, data_entry) -> List:
    def modify_document(self, text: str) -> str:
        text_in = RegexModifier.add_start_end_spaces(text)
        for regex_params in self.regex_params_list:
            text_out = re.sub(
                pattern=regex_params["pattern"],
                repl=regex_params["repl"],
                string=text_in,
                # note: this count param is the maximum number of pattern occurrences to be replaced.
                count=regex_params.get("count", 0),
            )
            text_in = text_out

        text_out = RegexModifier.remove_extra_spaces(text_out)

        return text_out

    @staticmethod
    def remove_extra_spaces(input_string):
        """
        Removes extra spaces in between words and at the start and end
        of the string.
        e.g. "abc  xyz   abc xyz" --> "abc xyz abc xyz"
        e.g. " abc xyz " --> "abc xyz"
        """
        output_string = " ".join(input_string.split())
        return output_string

    @staticmethod
    def add_start_end_spaces(input_string):
        """
        Adds spaces at the start and end of the input string.
        This is useful for when we specify we are looking for a particular
        word " <word> ". This will ensure we will find the word even
        if it is at the beginning or end of the utterances (ie. there will
        definitely be two spaces around the word).

        e.g. "abc xyz" --> " abc xyz "
        """
        # ensure no extra spaces
        no_extra_spaces_string = RegexModifier.remove_extra_spaces(input_string)
        output_string = f" {no_extra_spaces_string} "

        return output_string
