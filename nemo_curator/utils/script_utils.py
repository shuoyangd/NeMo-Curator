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
import os


class ArgumentHelper:
    """
    A helper class to add common arguments to an argparse.ArgumentParser instance.
    """

    def __init__(self, parser: argparse.ArgumentParser):
        self.parser = parser

    @staticmethod
    def attach_bool_arg(
        parser: argparse.ArgumentParser,
        flag_name: str,
        default: bool = False,
        help: str = None,
    ):
        attr_name = flag_name.replace("-", "_")
        help = flag_name.replace("-", " ") if help is None else help
        parser.add_argument(
            "--{}".format(flag_name),
            dest=attr_name,
            action="store_true",
            help=help,
        )
        parser.add_argument(
            "--no-{}".format(flag_name),
            dest=attr_name,
            action="store_false",
            help=help,
        )

        parser.set_defaults(**{attr_name: default})

    def add_arg_batch_size(
        self,
        default: int = 64,
        help: str = "Number of files to read into memory at a time.",
    ):
        self.parser.add_argument(
            "--batch-size",
            type=int,
            default=default,
            help=help,
        )

    def add_arg_device(self):
        self.parser.add_argument(
            "--device",
            type=str,
            default="gpu",
            help="Device to run the script on. Either 'cpu' or 'gpu'.",
        )

    def add_arg_enable_spilling(self):
        self.parser.add_argument("--enable-spilling", action="store_true")

    def add_arg_language(self, help: str):
        self.parser.add_argument(
            "--language",
            type=str,
            default="en",
            help=help,
        )

    def add_arg_log_dir(self, default: str):
        self.parser.add_argument(
            "--log-dir",
            type=str,
            default=default,
            help="The output log directory where node and local"
            " ranks will write their respective log files",
        )

    def add_arg_input_data_dir(
        self,
        required=False,
        help: str = "Input directory consisting of .jsonl files that are accessible "
        "to all nodes. Use this for a distributed file system",
    ):
        self.parser.add_argument(
            "--input-data-dir",
            type=str,
            default=None,
            required=required,
            help=help,
        )

    def add_arg_input_file_type(
        self,
        choices=None,
        required=False,
        help="File type of the dataset to be read in. Supported file formats "
        "include 'jsonl' (default), 'pickle', or 'parquet'.",
    ):
        self.parser.add_argument(
            "--input-file-type",
            type=str,
            default="jsonl",
            required=required,
            choices=choices,
            help=help,
        )

    def add_arg_input_file_extension(
        self,
        help: str = "The file extension of the input files. If not provided, the input file type will be used.",
    ):
        self.parser.add_argument(
            "--input-file-extension",
            type=str,
            default=None,
            help=help,
        )

    def add_arg_input_local_data_dir(self):
        self.parser.add_argument(
            "--input-local-data-dir",
            type=str,
            default=None,
            help="Input directory consisting of dataset files. "
            "Use this argument when a distributed file system is not available.",
        )

    def add_arg_input_meta(self):
        self.parser.add_argument(
            "--input-meta",
            type=str,
            default=None,
            help="A string formatted as a dictionary, which outlines the field names and "
            "their respective data types within the JSONL input files.",
        )

    def add_arg_input_text_field(self):
        self.parser.add_argument(
            "--input-text-field",
            type=str,
            default="text",
            help="The name of the field within each datapoint object of the input "
            "file that contains the text.",
        )

    def add_arg_minhash_length(self):
        self.parser.add_argument(
            "--minhash-length",
            type=int,
            default=260,
            help="The minhash signature length of each input document",
        )

    def add_arg_nvlink_only(self):
        self.parser.add_argument(
            "--nvlink-only",
            action="store_true",
            help="Start a local cluster with only NVLink enabled."
            "Only applicable when protocol=ucx and no scheduler file/address is specified",
        )

    def add_arg_output_data_dir(self, help: str):
        self.parser.add_argument(
            "--output-data-dir",
            type=str,
            required=True,
            help=help,
        )

    def add_arg_output_dir(
        self, required=False, help: str = "The output directory to write results in"
    ):
        self.parser.add_argument(
            "--output-dir",
            type=str,
            required=required,
            help=help,
        )

    def add_arg_output_file_type(
        self,
        choices=None,
        help="File type the dataset will be written to. Supported file formats "
        "include 'jsonl' (default), 'pickle', or 'parquet'.",
    ):
        self.parser.add_argument(
            "--output-file-type",
            type=str,
            default="jsonl",
            choices=choices,
            help=help,
        )

    def add_arg_output_train_file(self, help: str, default: str = None):
        self.parser.add_argument(
            "--output-train-file",
            type=str,
            default=default,
            help=help,
        )

    def add_arg_protocol(self):
        self.parser.add_argument(
            "--protocol",
            type=str,
            default="ucx",
            help="Protcol to use for dask cluster. "
            "Note: This only applies to the localCUDACluster. If providing an user created "
            "cluster refer to "
            "https://docs.rapids.ai/api/dask-cuda/stable/api.html#cmdoption-dask-cuda-protocol",  # noqa: E501
        )

    def add_arg_rmm_pool_size(self):
        self.parser.add_argument(
            "--rmm-pool-size",
            type=str,
            default="14GB",
            help="Initial pool size to use for the RMM Pool Memory allocator. "
            "Note: This only applies to the localCUDACluster. If providing an user created "
            "cluster refer to "
            "https://docs.rapids.ai/api/dask-cuda/stable/api.html#cmdoption-dask-cuda-rmm-pool-size",  # noqa: E501
        )

    def add_arg_scheduler_address(self):
        self.parser.add_argument(
            "--scheduler-address",
            type=str,
            default=None,
            help="Address to the scheduler of a created dask cluster. If not provided"
            "a single node LocalCUDACluster will be started.",
        )

    def add_arg_scheduler_file(self):
        self.parser.add_argument(
            "--scheduler-file",
            type=str,
            default=None,
            help="Path to the scheduler file of a created dask cluster. If not provided"
            " a single node LocalCUDACluster will be started.",
        )

    def add_arg_seed(
        self,
        default=42,
        help: str = "If specified, the random seed used for shuffling.",
    ):
        self.parser.add_argument(
            "--seed",
            type=int,
            default=default,
            help=help,
        )

    def add_arg_set_torch_to_use_rmm(self):
        self.parser.add_argument("--set-torch-to-use-rmm", action="store_true")

    def add_arg_shuffle(self, help: str):
        ArgumentHelper.attach_bool_arg(
            self.parser,
            "shuffle",
            help=help,
        )

    def add_arg_text_ddf_blocksize(self):
        self.parser.add_argument(
            "--text-ddf-blocksize",
            type=int,
            default=256,
            help="The block size for chunking jsonl files for text ddf in mb",
        )

    def add_arg_model_path(self, help="The path to the model file"):
        self.parser.add_argument(
            "--pretrained-model-name-or-path",
            type=str,
            help=help,
            required=False,
        )

    def add_arg_autocaset(self, help="Whether to use autocast or not"):
        ArgumentHelper.attach_bool_arg(
            parser=self.parser,
            flag_name="autocast",
            default=True,
            help=help,
        )

    def add_distributed_args(self) -> argparse.ArgumentParser:
        """
        Adds default set of arguments that are needed for Dask cluster setup
        """
        self.parser.add_argument(
            "--device",
            type=str,
            default="cpu",
            help="Device to run the script on. Either 'cpu' or 'gpu'.",
        )
        self.parser.add_argument(
            "--files-per-partition",
            type=int,
            default=2,
            help="Number of jsonl files to combine into single partition",
        )
        self.parser.add_argument(
            "--n-workers",
            type=int,
            default=os.cpu_count(),
            help="The number of workers to run in total on the Dask CPU cluster",
        )
        self.parser.add_argument(
            "--num-files",
            type=int,
            default=None,
            help="Upper limit on the number of json files to process",
        )
        self.parser.add_argument(
            "--nvlink-only",
            action="store_true",
            help="Start a local cluster with only NVLink enabled."
            "Only applicable when protocol=ucx and no scheduler file/address is specified",
        )
        self.parser.add_argument(
            "--protocol",
            type=str,
            default="tcp",
            help="Protcol to use for dask cluster"
            "Note: This only applies to the localCUDACluster. If providing an user created "
            "cluster refer to"
            "https://docs.rapids.ai/api/dask-cuda/stable/api.html#cmdoption-dask-cuda-protocol",  # noqa: E501
        )
        self.parser.add_argument(
            "--rmm-pool-size",
            type=str,
            default=None,
            help="Initial pool size to use for the RMM Pool Memory allocator"
            "Note: This only applies to the LocalCUDACluster. If providing an user created "
            "cluster refer to"
            "https://docs.rapids.ai/api/dask-cuda/stable/api.html#cmdoption-dask-cuda-rmm-pool-size",  # noqa: E501
        )
        self.parser.add_argument(
            "--scheduler-address",
            type=str,
            default=None,
            help="Address to the scheduler of a created dask cluster. If not provided"
            "a single node Cluster will be started.",
        )
        self.parser.add_argument(
            "--scheduler-file",
            type=str,
            default=None,
            help="Path to the scheduler file of a created dask cluster. If not provided"
            " a single node Cluster will be started.",
        )
        self.parser.add_argument(
            "--threads-per-worker",
            type=int,
            default=1,
            help="The number of threads ot launch per worker on the Dask CPU cluster. Usually best set at 1 due to the GIL.",
        )

        return self.parser

    @staticmethod
    def parse_client_args(args: argparse.Namespace):
        """
        Extracts relevant arguments from an argparse namespace to pass to get_client
        """
        relevant_args = [
            "scheduler_address",
            "scheduler_file",
            "n_workers",
            "threads_per_worker",
            "nvlink_only",
            "protocol",
            "rmm_pool_size",
            "enable_spilling",
            "set_torch_to_use_rmm",
        ]
        dict_args = vars(args)

        parsed_args = {arg: dict_args[arg] for arg in relevant_args if arg in dict_args}
        if "device" in dict_args:
            parsed_args["cluster_type"] = dict_args["device"]

        return parsed_args

    @staticmethod
    def parse_distributed_classifier_args(
        description="Default distributed classifier argument parser",
    ) -> argparse.ArgumentParser:
        """
        Adds default set of arguments that are common to multiple stages
        of the pipeline
        """
        parser = argparse.ArgumentParser(
            description,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        argumentHelper = ArgumentHelper(parser)
        argumentHelper.add_distributed_args()
        argumentHelper.add_arg_input_data_dir(required=True)
        argumentHelper.add_arg_output_data_dir(help="The path of the output files")
        argumentHelper.add_arg_input_file_type()
        argumentHelper.add_arg_input_file_extension()
        argumentHelper.add_arg_output_file_type()
        argumentHelper.add_arg_input_text_field()
        argumentHelper.add_arg_enable_spilling()
        argumentHelper.add_arg_set_torch_to_use_rmm()
        argumentHelper.add_arg_batch_size(
            help="The batch size to be used for inference"
        )
        argumentHelper.add_arg_model_path()
        argumentHelper.add_arg_autocaset()

        # Set low default RMM pool size for classifier
        # to allow pytorch to grow its memory usage
        # by default
        argumentHelper.parser.set_defaults(rmm_pool_size="512MB")
        # Setting to False makes it more stable for long running jobs
        # possibly because of memory fragmentation
        argumentHelper.parser.set_defaults(set_torch_to_use_rmm=False)
        return argumentHelper.parser

    @staticmethod
    def parse_gpu_dedup_args(description: str) -> argparse.ArgumentParser:
        """
        Adds default set of arguments that are common to multiple stages
        of the pipeline
        """

        parser = argparse.ArgumentParser(
            description,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        argumentHelper = ArgumentHelper(parser)

        argumentHelper.add_distributed_args()

        # Set default device to GPU for dedup
        argumentHelper.parser.set_defaults(device="gpu")
        argumentHelper.parser.set_defaults(set_torch_to_use_rmm=False)
        argumentHelper.parser.add_argument(
            "--input-data-dirs",
            type=str,
            nargs="+",
            default=None,
            help="Input directories consisting of .jsonl files that are accessible "
            "to all nodes. This path must be accessible by all machines in the cluster",
        )
        argumentHelper.parser.add_argument(
            "--input-json-text-field",
            type=str,
            default="text",
            help="The name of the field within each json object of the jsonl "
            "file that contains the text from which minhashes will be computed. ",
        )
        argumentHelper.parser.add_argument(
            "--input-json-id-field",
            type=str,
            default="adlr_id",
            help="The name of the field within each json object of the jsonl "
            "file that assigns a unqiue ID to each document. "
            "Can be created by running the script "
            "'./prospector/add_id.py' which adds the field 'adlr_id' "
            "to the documents in a distributed fashion",
        )
        argumentHelper.parser.add_argument(
            "--log-dir",
            type=str,
            default="./logs/",
            help="The output log directory where node and local",
        )
        argumentHelper.parser.add_argument(
            "--profile-path",
            type=str,
            default=None,
            help="Path to save dask profile",
        )

        return argumentHelper.parser

    @staticmethod
    def parse_semdedup_args(
        add_input_args=False,
        description="Default argument parser for semantic deduplication",
    ) -> argparse.ArgumentParser:
        """
        Adds default set of arguments that are common to multiple stages of the semantic deduplication pipeline
        of the pipeline
        """
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description=description,
        )
        argumentHelper = ArgumentHelper(parser)
        argumentHelper.add_distributed_args()
        if add_input_args:
            argumentHelper.add_arg_input_data_dir(required=True)
            argumentHelper.add_arg_input_file_extension()
            argumentHelper.add_arg_input_file_type()
            argumentHelper.add_arg_input_text_field()

        argumentHelper.parser.add_argument(
            "--config-file",
            type=str,
            help="Path to the semdedup config file",
            required=True,
        )
        # Set low default RMM pool size for classifier
        # to allow pytorch to grow its memory usage
        # by default
        parser.set_defaults(rmm_pool_size="512MB")
        parser.set_defaults(device="gpu")
        parser.set_defaults(set_torch_to_use_rmm=False)
        return parser
