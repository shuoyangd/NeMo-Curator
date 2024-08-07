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
import time

from nemo_curator import DomainClassifier
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.script_utils import ArgumentHelper


def main(args):
    global_st = time.time()

    # Input can be a string or list
    input_file_path = "/path/to/data"
    output_file_path = "./"

    client = get_client(**ArgumentHelper.parse_client_args(args))

    input_dataset = DocumentDataset.read_json(
        input_file_path, backend="cudf", add_filename=True
    )

    domain_classifier = DomainClassifier(filter_by=["Games", "Sports"])
    result_dataset = domain_classifier(dataset=input_dataset)

    result_dataset.to_json(output_file_dir=output_file_path, write_to_filename=True)

    global_et = time.time()
    print(
        f"Total time taken for domain classifier inference: {global_et-global_st} s",
        flush=True,
    )

    client.close()


def attach_args(
    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    ),
):
    argumentHelper = ArgumentHelper(parser)

    argumentHelper.add_arg_device()
    argumentHelper.add_arg_enable_spilling()
    argumentHelper.add_arg_nvlink_only()
    argumentHelper.add_arg_protocol()
    argumentHelper.add_arg_rmm_pool_size()
    argumentHelper.add_arg_scheduler_address()
    argumentHelper.add_arg_scheduler_file()
    argumentHelper.add_arg_set_torch_to_use_rmm()

    return argumentHelper.parser


if __name__ == "__main__":
    main(attach_args().parse_args())