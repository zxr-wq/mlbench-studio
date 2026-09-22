"""统一模型注册方式（统一规范 第 5 节）。

用法::

    @register_model("knn")
    class KNN(BaseModel):
        ...

    @register_model("pca", implementation="sklearn")
    class PCASklearn(BaseModel):
        ...

模型统一通过 ModelFactory 创建（见 factory.py），不要在主程序里写 if/elif。
"""

# _REGISTRY[(name, implementation)] = class
_REGISTRY = {}

IMPLEMENTATIONS = ("scratch", "sklearn")


def register_model(name, implementation="scratch"):
    """把模型类注册到统一名称 name 上，按 implementation 区分自实现 / sklearn 对照。"""

    def decorator(cls):
        key = (name, implementation)
        if key in _REGISTRY:
            raise ValueError(f"模型 {key} 已被注册")
        cls.model_name = name
        cls.implementation = implementation
        _REGISTRY[key] = cls
        return cls

    return decorator


def get_model_class(name, implementation="scratch"):
    key = (name, implementation)
    if key not in _REGISTRY:
        raise KeyError(f"未注册的模型: name={name!r} implementation={implementation!r}")
    return _REGISTRY[key]


def registered_names(implementation="scratch"):
    """当前已注册的统一算法名（过滤掉其他人尚未完成的算法）。"""
    return sorted(name for (name, impl) in _REGISTRY if impl == implementation)
