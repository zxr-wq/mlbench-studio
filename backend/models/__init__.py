"""导入全部模型模块以触发 @register_model 注册。

以后其他人增加算法时，在对应目录添加文件并在这里 import 一次即可。
"""

from backend.models.scratch import kmeans as scratch_kmeans  # noqa: F401
from backend.models.scratch import pca as scratch_pca  # noqa: F401
from backend.models.scratch import svm as scratch_svm  # noqa: F401
from backend.models.sklearn import kmeans as sklearn_kmeans  # noqa: F401
from backend.models.sklearn import pca as sklearn_pca  # noqa: F401
from backend.models.sklearn import svm as sklearn_svm  # noqa: F401
