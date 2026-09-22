import type { ExperimentResult } from '../types/benchmark'

export interface AlgorithmItem {
  id: string
  name: string
  shortName: string
  family: '分类' | '聚类' | '降维' | '集成' | '神经网络'
  implementation: '自主实现' | 'Sklearn 对照'
  status: 'ready' | 'draft'
  description: string
  accent: string
  tags: string[]
}

export interface DatasetItem {
  id: string
  name: string
  task: string
  samples: number
  features: number
  classes: number
  description: string
  accent: string
}

export const algorithms: AlgorithmItem[] = [
  { id: 'svm', name: '支持向量机', shortName: 'SVM', family: '分类', implementation: '自主实现', status: 'ready', description: '寻找最大间隔分类超平面，支持线性与核方法。', accent: '#e46647', tags: ['Margin', 'Kernel'] },
  { id: 'decision_tree', name: '决策树', shortName: 'CART', family: '分类', implementation: '自主实现', status: 'ready', description: '通过递归划分特征空间构造可解释的树结构。', accent: '#34715a', tags: ['Gini', 'Tree'] },
  { id: 'knn', name: 'K 近邻', shortName: 'KNN', family: '分类', implementation: '自主实现', status: 'ready', description: '根据特征空间中最近的训练样本完成预测。', accent: '#4f73b5', tags: ['Distance', 'Voting'] },
  { id: 'naive_bayes', name: '朴素贝叶斯', shortName: 'NB', family: '分类', implementation: '自主实现', status: 'ready', description: '利用条件独立假设得到高效概率分类器。', accent: '#885e9f', tags: ['Bayes', 'Probability'] },
  { id: 'kmeans', name: 'K-Means', shortName: 'KM', family: '聚类', implementation: '自主实现', status: 'ready', description: '迭代优化簇内平方误差的经典聚类算法。', accent: '#357b8a', tags: ['Cluster', 'Centroid'] },
  { id: 'pca', name: '主成分分析', shortName: 'PCA', family: '降维', implementation: '自主实现', status: 'ready', description: '用最大方差方向构造低维正交表示。', accent: '#656e70', tags: ['SVD', 'Projection'] },
  { id: 'random_forest', name: '随机森林', shortName: 'RF', family: '集成', implementation: '自主实现', status: 'ready', description: '结合 Bagging 与随机特征子集的树模型集成。', accent: '#58723d', tags: ['Bagging', 'Forest'] },
  { id: 'gradient_boosting', name: '梯度提升', shortName: 'GB', family: '集成', implementation: '自主实现', status: 'ready', description: '逐轮拟合残差并累加弱学习器。', accent: '#a15454', tags: ['Ensemble', 'Loss'] },
]

export const datasets: DatasetItem[] = [
  { id: 'breast_cancer', name: 'Breast Cancer Wisconsin', task: '二分类', samples: 569, features: 30, classes: 2, description: '来自乳腺肿块细胞核图像的诊断特征，适合二分类模型比较。', accent: '#e8c7bd' },
  { id: 'iris', name: 'Iris', task: '多分类', samples: 150, features: 4, classes: 3, description: '经典鸢尾花数据集，适合展示决策边界和基础分类流程。', accent: '#c8dbcd' },
  { id: 'wine', name: 'Wine', task: '多分类', samples: 178, features: 13, classes: 3, description: '葡萄酒化学分析数据，用于比较尺度敏感模型和树模型。', accent: '#e3d2b5' },
  { id: 'digits', name: 'Optical Digits', task: '多分类', samples: 1797, features: 64, classes: 10, description: '8×8 手写数字灰度特征，适合高维分类与降维实验。', accent: '#cbd2e3' },
]

export const initialRuns: ExperimentResult[] = [
  { id: 'EXP-024', model: 'SVM', dataset: 'Breast Cancer', accuracy: 0.972, f1: 0.968, duration: 1.82, status: 'completed', createdAt: '今天 15:42' },
  { id: 'EXP-023', model: 'Random Forest', dataset: 'Wine', accuracy: 0.961, f1: 0.958, duration: 2.34, status: 'completed', createdAt: '今天 14:18' },
  { id: 'EXP-022', model: 'Decision Tree', dataset: 'Iris', accuracy: 0.947, f1: 0.944, duration: 0.46, status: 'completed', createdAt: '昨天 20:06' },
  { id: 'EXP-021', model: 'KNN', dataset: 'Wine', accuracy: 0.938, f1: 0.934, duration: 0.12, status: 'completed', createdAt: '昨天 18:34' },
  { id: 'EXP-020', model: 'Naive Bayes', dataset: 'Breast Cancer', accuracy: 0.921, f1: 0.916, duration: 0.08, status: 'completed', createdAt: '9 月 7 日' },
  { id: 'EXP-019', model: 'Logistic Regression', dataset: 'Iris', accuracy: 0.953, f1: 0.951, duration: 0.31, status: 'completed', createdAt: '9 月 7 日' },
]
