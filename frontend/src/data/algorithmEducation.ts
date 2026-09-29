export interface AlgorithmEducation {
  principle: string
  useWhen: string
  visuals: string[]
}

export const algorithmEducation: Record<string, AlgorithmEducation> = {
  linear_regression: { principle: '用特征的线性组合拟合连续目标，并让预测误差的平方和尽量小。', useWhen: '目标是连续数值，例如疾病进展、房价或销量。', visuals: ['真实值 vs 预测值散点图', '残差分布直方图', '绝对系数贡献条形图'] },
  logistic_regression: { principle: '将线性得分映射为各类别的概率，再选择概率最高的类别。', useWhen: '需要可解释、训练快的基础分类器。', visuals: ['混淆矩阵', '真实/预测类别分布', '二维测试集投影', '绝对系数贡献', '训练损失曲线'] },
  knn: { principle: '预测时寻找训练集中距离最近的 K 个样本，让邻居投票决定类别。', useWhen: '样本量不大、相似样本通常属于同一类时。', visuals: ['混淆矩阵', '真实/预测类别分布', '二维测试集投影'] },
  naive_bayes: { principle: '根据贝叶斯公式计算类别概率，并假设给定类别后各特征相互独立。', useWhen: '需要非常快的概率基线模型时。', visuals: ['混淆矩阵', '真实/预测类别分布', '二维测试集投影'] },
  svm: { principle: '寻找类别之间间隔最大的分隔边界；核函数可表达非线性边界。', useWhen: '特征维度较高、类别边界较清晰的中小型数据集。', visuals: ['混淆矩阵', '真实/预测类别分布', '二维测试集投影'] },
  decision_tree: { principle: '反复选择最能降低类别混杂的特征与阈值，形成一系列可读规则。', useWhen: '希望模型规则直观、无需特征缩放时。', visuals: ['混淆矩阵', '二维测试集投影', '树分裂特征使用次数'] },
  random_forest: { principle: '训练多棵看到不同样本和特征的决策树，再让它们投票。', useWhen: '需要较稳健的非线性分类表现时。', visuals: ['混淆矩阵', '真实/预测类别分布', '二维测试集投影', '森林特征使用统计'] },
  gradient_boosting: { principle: '后一轮浅树专门修正前一轮的错误，逐步把多个弱学习器叠加。', useWhen: '需要较强的表格数据拟合能力，且可接受更长训练时间时。', visuals: ['混淆矩阵', '二维测试集投影', '提升树分裂特征统计', '训练损失曲线'] },
  kmeans: { principle: '不断将样本分给最近质心并更新质心，使每个簇内部尽量紧凑。', useWhen: '没有标签，希望探索数据天然分组时。', visuals: ['聚类二维投影', '每个簇的样本规模'] },
  pca: { principle: '寻找方差最大的正交方向，用较少主成分保留尽可能多的信息。', useWhen: '想压缩维度、观察高维数据结构或作为后续模型预处理时。', visuals: ['主成分解释方差谱', '前两个主成分投影'] },
}

export const metricEducation = {
  classification: [
    ['Accuracy / F1', '越高越好。类别不平衡时，F1 通常比 Accuracy 更可靠。'],
    ['Precision', '越高越好。它高表示“预测为某类”的样本更可信。'],
    ['Recall', '越高越好。它高表示该类别的真实样本更少被漏掉。'],
  ],
  regression: [
    ['R²', '越接近 1 越好；0 表示不优于直接预测平均值，负数表示更差。'],
    ['MSE / RMSE / MAE', '越低越好；RMSE 与目标同单位，MAE 更不受极端误差影响。'],
  ],
  clustering: [
    ['Silhouette', '范围约为 -1 到 1，越接近 1 越说明簇内紧、簇间分离。'],
    ['Inertia', '越低簇内越紧凑，但只应在同一数据集、同一 K 下比较。'],
  ],
  dimensionality_reduction: [
    ['累计解释方差', '越高表示保留的信息越多；常用目标是 80%–95%，也要结合降维目的判断。'],
  ],
} as const

export function modelKey(name: string) {
  return name.toLowerCase().replaceAll(' ', '_')
}
