<div align="center">

  <a href="https://github.com/NVIDIA/NeMo-Curator/blob/main/LICENSE">![https://pypi.org/project/nemo-curator](https://img.shields.io/github/license/NVIDIA/NeMo-Curator)</a>
  <a href="https://pypi.org/project/nemo-curator/">![https://pypi.org/project/nemo-curator/](https://img.shields.io/pypi/pyversions/nemo-curator.svg)</a>
  <a href="https://github.com/NVIDIA/NeMo-Curator/graphs/contributors">![NVIDIA/NeMo-Curator](https://img.shields.io/github/contributors/NVIDIA/NeMo-Curator)</a>
  <a href="https://github.com/NVIDIA/NeMo-Curator/releases">![https://github.com/NVIDIA/NeMo-Curator/releases](https://img.shields.io/github/release/NVIDIA/NeMo-Curator)</a>
  <a href="https://pypi.org/project/nemo-curator/">![https://github.com/Naereen/badges/](https://badgen.net/badge/open%20source/❤/blue?icon=github)</a>

</div>

# NeMo Curator
🚀 **The GPU-Accelerated Open Source Framework for Efficient Large Language Model Data Curation** 🚀

<p align="center">
  <img src="./docs/user-guide/images/diagram.png" alt="diagram"/>
</p>

NeMo Curator is a Python library specifically designed for fast and scalable dataset preparation and curation for [large language model (LLM)](https://www.nvidia.com/en-us/glossary/large-language-models/) use-cases such as foundation model pretraining, domain-adaptive pretraining (DAPT), supervised fine-tuning (SFT) and paramter-efficient fine-tuning (PEFT). It greatly accelerates data curation by leveraging GPUs with [Dask](https://www.dask.org/) and [RAPIDS](https://developer.nvidia.com/rapids), resulting in significant time savings. The library provides a customizable and modular interface, simplifying pipeline expansion and accelerating model convergence through the preparation of high-quality tokens.

At the core of the NeMo Curator is the `DocumentDataset` which serves as the the main dataset class. It acts as a straightforward wrapper around a Dask `DataFrame`. The Python library offers easy-to-use methods for expanding the functionality of your curation pipeline while eliminating scalability concerns.

## Key Features

NeMo Curator provides a collection of scalable data-mining modules. Some of the key features include:

- [Data download and text extraction](docs/user-guide/download.rst)

  - Default implementations for downloading and extracting Common Crawl, Wikipedia, and ArXiv data
  - Easily customize the download and extraction and extend to other datasets

- [Language identification and separation](docs/user-guide/languageidentificationunicodeformatting.rst) with [fastText](https://fasttext.cc/docs/en/language-identification.html) and [pycld2](https://pypi.org/project/pycld2/)

- [Text reformatting and cleaning](docs/user-guide/languageidentificationunicodeformatting.rst) to fix unicode decoding errors via [ftfy](https://ftfy.readthedocs.io/en/latest/)

- [Quality filtering](docs/user-guide/qualityfiltering.rst)

  - Multilingual heuristic-based filtering
  - Classifier-based filtering via [fastText](https://fasttext.cc/)

- [Document-level deduplication](docs/user-guide/gpudeduplication.rst)

  - exact and fuzzy (near-identical) deduplication are accelerated using cuDF and Dask
  - For fuzzy deduplication, our implementation follows the method described in [Microsoft Turing NLG 530B](https://arxiv.org/abs/2201.11990)
  - For semantic deduplication,  our implementation follows the method described in [SemDeDup](https://arxiv.org/pdf/2303.09540) by Meta AI (FAIR) [facebookresearch/SemDeDup](https://github.com/facebookresearch/SemDeDup)

- [Multilingual downstream-task decontamination](docs/user-guide/taskdecontamination.rst) following the approach of [OpenAI GPT3](https://arxiv.org/pdf/2005.14165.pdf) and [Microsoft Turing NLG 530B](https://arxiv.org/abs/2201.11990)

- [Distributed data classification](docs/user-guide/distributeddataclassification.rst)

  - Multi-node, multi-GPU classifier inference
  - Provides sophisticated domain and quality classification
  - Flexible interface for extending to your own classifier network

- [Personal identifiable information (PII) redaction](docs/user-guide/personalidentifiableinformationidentificationandremoval.rst) for removing addresses, credit card numbers, social security numbers, and more

These modules offer flexibility and permit reordering, with only a few exceptions. In addition, the [NeMo Framework Launcher](https://github.com/NVIDIA/NeMo-Megatron-Launcher) provides pre-built pipelines that can serve as a foundation for your customization use cases.

## Resources

- [Documentation](docs/)
- [Examples](examples/)
- [Tutorials](tutorials/)
- Blog posts
  - [Curating Trillion-Token Datasets: Introducing NVIDIA NeMo Data Curator](https://developer.nvidia.com/blog/curating-trillion-token-datasets-introducing-nemo-data-curator/)
  - [Scale and Curate High-Quality Datasets for LLM Training with NVIDIA NeMo Curator](https://developer.nvidia.com/blog/scale-and-curate-high-quality-datasets-for-llm-training-with-nemo-curator/)
  - [Curating Custom Datasets for LLM Training with NVIDIA NeMo Curator](https://developer.nvidia.com/blog/curating-custom-datasets-for-llm-training-with-nvidia-nemo-curator/)
  - [Curating Custom Datasets for LLM Parameter-Efficient Fine-Tuning with NVIDIA NeMo Curator](https://developer.nvidia.com/blog/curating-custom-datasets-for-llm-parameter-efficient-fine-tuning-with-nvidia-nemo-curator/)

## Get Started

This section explains how to install NeMo Curator and use the Python library, Python modules, and CLI scripts. It also includes a list of tutorials to help you get started right away. Finally, this section explains how to use the NeMo Framework Launcher as an alternative method for interfacing with NeMo Curator.

### Install NeMo Curator

#### Requirements

Before installing NeMo Curator, ensure that the following requirements are met:

- Python 3.10
- Ubuntu 22.04/20.04
- NVIDIA GPU (optional)
  - Volta™ or higher ([compute capability 7.0+](https://developer.nvidia.com/cuda-gpus))
  - CUDA 12 (or above)

You can install NeMo-Curator from PyPi, from source or get it through the NeMo Framework container.

#### From PyPi

To install the CPU-only modules:

```bash
pip install nemo-curator
```

To install the CPU and CUDA-accelerated modules:

```bash
pip install --extra-index-url https://pypi.nvidia.com nemo-curator[cuda12x]
```

#### From Source

1. Clone the NeMo Curator repository in GitHub.

    ```bash
    git clone https://github.com/NVIDIA/NeMo-Curator.git
    cd NeMo-Curator
    ```

2. Install the modules that you need.

    To install the CPU-only modules:

    ```bash
    pip install .
    ```

    To install the CPU and CUDA-accelerated modules:

    ```bash
    pip install --extra-index-url https://pypi.nvidia.com ".[cuda12x]"
    ```

#### From the NeMo Framework Container

The latest release of NeMo Curator comes preinstalled in the [NeMo Framework Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags). If you want the latest commit inside the container, you can reinstall NeMo Curator using:

```bash
pip uninstall nemo-curator
rm -r /opt/NeMo-Curator
git clone https://github.com/NVIDIA/NeMo-Curator.git /opt/NeMo-Curator
pip install --extra-index-url https://pypi.nvidia.com /opt/NeMo-Curator[cuda12x]
```
And follow the instructions for installing from source from [above](#from-source).

## Use NeMo Curator
### Python API Quick Example

The following snippet demonstrates how to create a small data curation pipeline that downloads and curates a small subset of the Common Crawl dataset.

```Python
# Download your dataset
dataset = download_common_crawl("/datasets/common_crawl/", "2021-04", "2021-10", url_limit=10)
# Build your pipeline
curation_pipeline = Sequential([
  # Fix unicode
  Modify(UnicodeReformatter()),
  # Discard short records
  ScoreFilter(WordCountFilter(min_words=80)),
  # Discard low-quality records
  ScoreFilter(FastTextQualityFilter(model_path="model.bin")),
  # Discard records from the evaluation metrics to prevent test set leakage.
  TaskDecontamination([Winogrande(), Squad(), TriviaQA()])
])
# Execute the pipeline on your dataset
curated_dataset = curation_pipeline(dataset)
```

### Explore NeMo Curator Tutorials

To get started with NeMo Curator, you can follow the tutorials [available here](https://github.com/NVIDIA/NeMo-Curator/tree/main/tutorials). These tutorials include:

- [`tinystories`](https://github.com/NVIDIA/NeMo-Curator/tree/main/tutorials/tinystories) which focuses on data curation for training LLMs from scratch.
- [`peft-curation`](https://github.com/NVIDIA/NeMo-Curator/tree/main/tutorials/peft-curation) which focuses on data curation for LLM parameter-efficient fine-tuning (PEFT) use-cases.
- [`distributed_data_classification`](https://github.com/NVIDIA/NeMo-Curator/tree/main/tutorials/distributed_data_classification) which focuses on using the quality and domain classifiers to help with data annotation.
- [`single_node_tutorial`](https://github.com/NVIDIA/NeMo-Curator/tree/main/tutorials/single_node_tutorial) which demonstrates an end-to-end data curation pipeline for curating Wikipedia data in Thai.


### Access Python Modules

The NeMo Curator section of the [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/datacuration/index.html) provides in-depth information about how the Python modules work. The [examples](examples/) directory in the GitHub repository provides scripts that showcase these modules.

### Use CLI Scripts

NeMo Curator also offers CLI scripts for you to use. The scripts in `nemo_curator/scripts` map closely to the supplied Python modules. Refer to the [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/datacuration/index.html) for more information about the Python modules and scripts.

### Use NeMo Framework Launcher

As an alternative method for interfacing with NeMo Curator, you can use the [NeMo Framework Launcher](https://github.com/NVIDIA/NeMo-Megatron-Launcher). The launcher enables you to easily configure the parameters and cluster. It can also automatically generate the SLURM batch scripts that wrap around the CLI scripts required to run your pipeline.

In addition, other methods are available to run NeMo Curator on SLURM. For example, refer to the example scripts in [`examples/slurm`](examples/slurm/) for information on how to run NeMo Curator on SLURM without the NeMo Framework Launcher.

## Module Ablation and Compute Performance

The modules within NeMo Curator were primarily designed to curate high-quality documents from Common Crawl snapshots in a scalable manner. To evaluate the quality of the curated Common Crawl documents, we conducted a series of ablation experiments. In these experiments, we trained a 357M-parameter GPT-style model using datasets generated at various stages of our data curation pipeline, which was implemented in NeMo Curator.

The following figure shows that the use of different data curation modules implemented in NeMo Curator led to improved model zero-shot downstream task performance.

<p align="center">
  <img src="./docs/user-guide/images/zeroshot_ablations.png" alt="drawing" width="700"/>
</p>

In terms of scalability and compute performance, using the combination of RAPIDS and Dask fuzzy deduplication enabled us to deduplicate the 1.1 Trillion token Red Pajama dataset in 1.8 hours with 64 NVIDIA A100 Tensor Core GPUs.

Additionally, using the CPU-based modules, the following table shows the time required and resulting data size reduction for each processing step [Common Crawl snapshot from November/December of 2020](https://commoncrawl.org/2020/12/nov-dec-2020-crawl-archive-now-available/) using 30 CPU nodes (with hardware similar to the `c5.24xlarge` [Amazon AWS C5 instance](https://aws.amazon.com/ec2/instance-types/c5/)).


<table>
  <thead>
    <tr>
      <th style="text-align:center">Dataset </th>
      <th colspan="2"> Download and text extraction</th>
      <th colspan="2">Text cleaning </th>
      <th colspan="2">Quality filtering</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td></td>
      <td>Time  </td>
      <td> Output Size </td>
      <td>Time </td>
      <td> Output Size </td>
      <td>Time </td>
      <td> Output Size </td>
    </tr>
    <tr>
      <td>Common Crawl 2020-50</td>
      <td> 36 hrs</td>
      <td>2.8 TB</td>
      <td> 1 hr </td>
      <td> 2.8 TB </td>
      <td> 0.2 hr </td>
      <td> 0.52 TB </td>
    </tr>
  </tbody>
</table>


## Contribute to NeMo Curator

We welcome community contributions! Please refer to [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for the process.
