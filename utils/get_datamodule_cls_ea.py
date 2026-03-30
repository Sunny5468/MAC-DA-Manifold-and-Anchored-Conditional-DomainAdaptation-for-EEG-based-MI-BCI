from datamodules.bcic4_2a_ea import (
    BCICIV2a, BCICIV2aLOSO, BCICIV2aLOSO_CDAN,
)
from datamodules.weibo2014_ea import (
    Weibo2014SubDep, Weibo2014LOSO, Weibo2014LOSO_CDAN,
)


def get_datamodule_cls(dataset_name: str):
    """
    Return the EA-capable DataModule class by dataset name.

    Args:
        dataset_name:
            - "bcic2a" / "bcic2a_loso" / "bcic2a_loso_cdan"
            - "weibo2014" / "weibo2014_loso" / "weibo2014_loso_cdan"
            (trailing "_ea" suffix is stripped automatically for output naming)
    """
    dataset_name = dataset_name.lower()

    # Strip explicit EA suffix used only for output directory naming.
    if dataset_name.endswith("_ea"):
        dataset_name = dataset_name[:-3]

    registry = {
        "bcic2a": BCICIV2a,
        "bcic2a_loso": BCICIV2aLOSO,
        "bcic2a_loso_cdan": BCICIV2aLOSO_CDAN,
        "weibo2014": Weibo2014SubDep,
        "weibo2014_loso": Weibo2014LOSO,
        "weibo2014_loso_cdan": Weibo2014LOSO_CDAN,
    }

    if dataset_name in registry:
        return registry[dataset_name]

    raise ValueError(
        f"Unknown dataset: {dataset_name}. "
        f"Available options: {', '.join(registry.keys())}"
    )
