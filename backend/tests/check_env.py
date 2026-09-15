"""环境自检：确认 numpy / sklearn 装好、iris 能加载、你的算法文件能被找到

运行方式（在 E:\\mlbench-studio 目录下）：
    .\\.venv\\Scripts\\python.exe backend\\tests\\check_env.py
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    import numpy as np
    import sklearn
    from sklearn.datasets import load_iris

    print("python  :", sys.version.split()[0])
    print("numpy   :", np.__version__)
    print("sklearn :", sklearn.__version__)

    data = load_iris()
    print("iris    :", data.data.shape, "特征:", list(data.feature_names))

    from models.scratch.decision_tree import DecisionTree

    tree = DecisionTree(max_depth=3)
    print("算法文件: 导入成功")
    print("task_type:", tree.task_type)
    print("get_params():", tree.get_params())
    print("\n环境一切正常，可以开始填 TODO 了。")


if __name__ == "__main__":
    main()
