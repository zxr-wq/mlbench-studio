"""统一模型基础接口（统一规范 第 1 节）。

所有算法必须继承 BaseModel 并声明 task_type。
因为十种算法包含分类、回归、聚类和降维，所以不强制所有算法完全一样：

- 分类 / 回归：额外实现 predict(X)
- 聚类（kmeans）：额外实现 predict(X) / fit_predict(X)
- 降维（pca）：额外实现 transform(X) / fit_transform(X)，不实现 predict
"""

TASK_TYPES = ("classification", "regression", "clustering", "dimensionality_reduction")


class BaseModel:
    """统一基础类。

    注意 fit 统一写成 fit(self, X, y=None)：
    - 分类、回归需要 X, y
    - K-Means、PCA 只需要 X
    """

    task_type = None

    def fit(self, X, y=None):
        """训练模型。"""
        raise NotImplementedError

    def get_params(self):
        """返回模型参数。"""
        return {}

    def get_visualization_data(self):
        """返回前端可视化需要的数据，没有则返回空字典。"""
        return {}
