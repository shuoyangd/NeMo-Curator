#!/bin/bash
# =============================================================================
# NeMo Curator - NRT SLURM translate-then-curate dry-run smoke test
#
# Runs the bitext recipe on a source-only German MMLPC manifest:
#   translation dry-run -> bitext curation filters -> JSONL writer
#
# This validates NRT Slurm + Pyxis + SlurmRayClient + RayDataExecutor +
# translate-then-curate wiring without loading vLLM. The dry-run translation
# stage copies src to tgt while preserving the same reader/expander handoff used
# by the real LLM translation stage.
# =============================================================================

#SBATCH --job-name=curator-bitext-nrt-dryrun
#SBATCH --account=nemotron_speech_translate
#SBATCH --partition=batch_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/bitext_translate_nrt_dryrun_%j.log
#SBATCH --error=logs/bitext_translate_nrt_dryrun_%j.log

set -euo pipefail

if [[ -z "${CURATOR_DIR:-}" ]]; then
    if [[ -f "${PWD}/tutorials/text/bitext-cleaning/main.py" ]]; then
        CURATOR_DIR="${PWD}"
    elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/tutorials/text/bitext-cleaning/main.py" ]]; then
        CURATOR_DIR="${SLURM_SUBMIT_DIR}"
    else
        CURATOR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    fi
fi

MANIFEST_FILE="${MANIFEST_FILE:-/lustre/fsw/portfolios/llmservice/users/nkoluguri/projects/granary-v2/outputs/granary_v2/labelled_human_text/MMLPC/de/pcstrip_sharded_manifests/manifest_0.jsonl}"
RUN_ROOT="${RUN_ROOT:-${CURATOR_DIR}/runs/bitext_translate_nrt_dryrun_${SLURM_JOB_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/output}"
LOG_DIR="${LOG_DIR:-${CURATOR_DIR}/logs}"
CACHE_DIR="${CURATOR_CACHE_DIR:-${CURATOR_DIR}/cache/bitext_translate_nrt_dryrun}"

CONTAINER_IMAGE="${CONTAINER_IMAGE:-docker://nvcr.io#nvidia/vllm:26.05-py3}"
CONTAINER_PYTHON="${CONTAINER_PYTHON:-python3}"
SRC_FIELD="${SRC_FIELD:-pnc_text}"
SRC_LANG="${SRC_LANG:-de}"
TGT_LANG="${TGT_LANG:-en}"
MAX_ROWS="${MAX_ROWS:-8}"
ROWS_PER_TASK="${ROWS_PER_TASK:-2}"
WORKER_CONNECT_TIMEOUT_S="${WORKER_CONNECT_TIMEOUT_S:-900}"

if [[ ! -f "${MANIFEST_FILE}" ]]; then
    echo "Manifest file not found: ${MANIFEST_FILE}" >&2
    exit 1
fi
if [[ ! -f "${CURATOR_DIR}/tutorials/text/bitext-cleaning/main.py" ]]; then
    echo "Curator checkout not found: ${CURATOR_DIR}" >&2
    exit 1
fi

MANIFEST_DIR="$(dirname "${MANIFEST_FILE}")"
mkdir -p "${LOG_DIR}" "${RUN_ROOT}" "${OUTPUT_DIR}" "${CACHE_DIR}" "${RUN_ROOT}/ray_ports" "${CACHE_DIR}/pip"
ENROOT_ROOT="${ENROOT_ROOT:-${CACHE_DIR}/enroot}"
export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH:-${ENROOT_ROOT}/cache}"
export ENROOT_DATA_PATH="${ENROOT_DATA_PATH:-${ENROOT_ROOT}/data}"
export ENROOT_RUNTIME_PATH="${ENROOT_RUNTIME_PATH:-${RUN_ROOT}/enroot/runtime}"
export ENROOT_TEMP_PATH="${ENROOT_TEMP_PATH:-${RUN_ROOT}/enroot/tmp}"
export PARALLEL_HOME="${PARALLEL_HOME:-${RUN_ROOT}/parallel}"
export TMPDIR="${TMPDIR:-${RUN_ROOT}/tmp}"
mkdir -p \
    "${ENROOT_CACHE_PATH}" \
    "${ENROOT_DATA_PATH}" \
    "${ENROOT_RUNTIME_PATH}" \
    "${ENROOT_TEMP_PATH}" \
    "${PARALLEL_HOME}" \
    "${TMPDIR}"

CONTAINER_CMD="${RUN_ROOT}/container_command.sh"
CONTAINER_MOUNTS="${RUN_ROOT}:${RUN_ROOT},${CACHE_DIR}:${CACHE_DIR},${LOG_DIR}:${LOG_DIR},${CURATOR_DIR}:/opt/nemo-curator:ro,${MANIFEST_DIR}:${MANIFEST_DIR}:ro"
export RAY_PORT_BROADCAST_DIR="${RUN_ROOT}/ray_ports"
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}"
export HF_HOME="${HF_HOME:-${CACHE_DIR}/hf_home}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_DIR}/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${CACHE_DIR}/pip}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cat > "${CONTAINER_CMD}" <<EOF
#!/bin/bash
set -euo pipefail

export RAY_TMPDIR="/tmp/ray_\${SLURM_JOB_ID}"
export RAY_PORT_BROADCAST_DIR="${RAY_PORT_BROADCAST_DIR}"
export HF_HOME="${HF_HOME}"
export TORCH_HOME="${TORCH_HOME}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="/opt/nemo-curator:\${PYTHONPATH:-}"

PYTHON="\$(command -v "${CONTAINER_PYTHON}")"
if [[ -z "\${PYTHON}" || ! -x "\${PYTHON}" ]]; then
    echo "[\$(hostname)] missing executable Python for ${CONTAINER_PYTHON}" >&2
    exit 127
fi

"\${PYTHON}" -m pip install --upgrade \
    "ray[default,data]>=2.55.1" \
    cosmos-xenna==0.2.0 \
    pandas pyarrow loguru requests fsspec

"\${PYTHON}" -c 'import nemo_curator, pandas, pyarrow, ray; print("imports_ok", ray.__version__, nemo_curator.__file__)'
"\${PYTHON}" /opt/nemo-curator/tutorials/text/bitext-cleaning/main.py \
    --slurm \
    --input-jsonl "${MANIFEST_FILE}" \
    --src-field "${SRC_FIELD}" \
    --src-lang "${SRC_LANG}" \
    --tgt-lang "${TGT_LANG}" \
    --translate \
    --translation-dry-run \
    --translation-work-dir "${RUN_ROOT}/translation_work" \
    --output-dir "${OUTPUT_DIR}" \
    --output-mode overwrite \
    --max-rows "${MAX_ROWS}" \
    --rows-per-task "${ROWS_PER_TASK}" \
    --worker-connect-timeout-s "${WORKER_CONNECT_TIMEOUT_S}"
EOF
chmod +x "${CONTAINER_CMD}"

echo "NRT dry-run translate-then-curate"
echo "Manifest: ${MANIFEST_FILE}"
echo "Direction: ${SRC_LANG}->${TGT_LANG} (${SRC_FIELD})"
echo "Output: ${OUTPUT_DIR}"

srun \
    --ntasks-per-node=1 \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --container-workdir="/opt/nemo-curator" \
    /bin/bash "${CONTAINER_CMD}"

echo "DONE ${OUTPUT_DIR}"
