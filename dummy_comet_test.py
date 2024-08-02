from nemo_curator.modules.filter import JointScoreFilter
from nemo_curator.filters import COMETQualityEstimationFilter
from nemo_curator.utils.distributed_utils import get_client

from tests.test_filters import two_lists_to_parallel_dataset, all_equal


if __name__ == '__main__':
    dataset = two_lists_to_parallel_dataset(
        [
            "This sentence will be translated on the Chinese side.",
            "This sentence will have something irrelevant on the Chinese side.",
        ],
        [
            "这句话在中文一侧会被翻译。",
            "至尊戒，驭众戒；至尊戒，寻众戒；魔戒至尊引众戒，禁锢众戒黑暗中。",
        ]
    )

    client = get_client(n_workers=1, rmm_pool_size=None)
    filter_ = JointScoreFilter(COMETQualityEstimationFilter())
    filtered_data = filter_(dataset)
    client.close()
