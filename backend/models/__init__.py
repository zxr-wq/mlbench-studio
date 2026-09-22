"""导入全部模型模块以触发 @register_model 注册。

以后其他人增加算法时，在对应目录添加文件并在这里 import 一次即可。
"""

from backend.models.scratch import kmeans as scratch_kmeans  # noqa: F401
from backend.models.scratch import pca as scratch_pca  # noqa: F401
from backend.models.scratch import svm as scratch_svm  # noqa: F401
from backend.models.scratch import knn as scratch_knn  # noqa: F401
from backend.models.scratch import naive_bayes as scratch_naive_bayes  # noqa: F401
from backend.models.scratch import decision_tree as scratch_decision_tree  # noqa: F401
from backend.models.scratch import random_forest as scratch_random_forest  # noqa: F401
from backend.models.scratch import gradient_boosting as scratch_gradient_boosting  # noqa: F401
from backend.models.sklearn import kmeans as sklearn_kmeans  # noqa: F401
from backend.models.sklearn import pca as sklearn_pca  # noqa: F401
from backend.models.sklearn import svm as sklearn_svm  # noqa: F401
from backend.models.sklearn import classifiers as sklearn_classifiers  # noqa: F401
