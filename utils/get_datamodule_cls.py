from datamodules import (
    BCICIV2a, BCICIV2aLOSO, BCICIV2aLOSO_CDAN,
    Weibo2014SubDep, Weibo2014LOSO, Weibo2014LOSO_CDAN,
)


def get_datamodule_cls(dataset_name: str):
    """
    根据数据集名称返回对应的 DataModule 类
    
    Args:
        dataset_name: 数据集名称
            - "bcic2a": BCI Competition IV 2a (Subject-Dependent)
            - "bcic2a_loso": BCI Competition IV 2a (Leave-One-Subject-Out)
            - "bcic2a_loso_cdan": BCI Competition IV 2a (LOSO + CDAN)
            - "weibo2014": Weibo2014 (Subject-Dependent)
            - "weibo2014_loso": Weibo2014 (Leave-One-Subject-Out)
            - "weibo2014_loso_cdan": Weibo2014 (LOSO + CDAN/DANN/CORAL)
    
    Returns:
        DataModule 类
    """
    dataset_name = dataset_name.lower()
    
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
    
    raise ValueError(f"Unknown dataset: {dataset_name}. "
                     f"Available options: {', '.join(registry.keys())}")
