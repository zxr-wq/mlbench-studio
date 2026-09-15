import { aBenchmarkRows } from '../data/aBenchmark'

export const demoMode = import.meta.env.VITE_DEMO_MODE === 'true'

const datasets = [
  { id: 'iris', name: 'Iris 鸢尾花', domain: '植物分类', description: '以花萼和花瓣的 4 个特征识别 3 种鸢尾花。', accent: '#7157d9', samples: 150, features: 4, classes: 3, task_type: 'classification' },
  { id: 'wine', name: 'Wine 葡萄酒', domain: '化学鉴别', description: '根据 13 项化学指标判断葡萄酒产区。', accent: '#c45a47', samples: 178, features: 13, classes: 3, task_type: 'classification' },
  { id: 'breast_cancer', name: 'Breast Cancer', domain: '医学诊断', description: '以细胞核统计特征区分良性与恶性肿瘤。', accent: '#1c8b76', samples: 569, features: 30, classes: 2, task_type: 'classification' },
]

const specs = [
  ['linear_regression', '线性回归', 'Linear', '线性模型', 'regression'], ['logistic_regression', '逻辑回归', 'Logistic', '线性模型', 'classification'],
  ['knn', 'K 近邻', 'KNN', '实例学习', 'classification'], ['naive_bayes', '高斯朴素贝叶斯', 'GNB', '概率模型', 'classification'],
  ['svm', '支持向量机', 'SVM', '最大间隔', 'classification'], ['kmeans', 'K-Means', 'KM', '聚类', 'clustering'],
  ['pca', '主成分分析', 'PCA', '降维', 'dimensionality_reduction'], ['decision_tree', 'CART 决策树', 'CART', '树模型', 'classification'],
  ['random_forest', '随机森林', 'RF', '集成学习', 'classification'], ['gradient_boosting', '梯度提升', 'GBDT', '集成学习', 'classification'],
]

const algorithms = specs.map(([id, name, short_name, family, task_type]) => ({
  id, name, short_name, family, task_type,
  description: `${name} 的统一模型目录项。`, principle: '通过统一 fit 与任务专用接口接入实验流程。', formula: id,
  strengths: ['Scratch', '可复现'], available: ['knn', 'naive_bayes'].includes(id),
  implementations: ['knn', 'naive_bayes'].includes(id) ? ['scratch', 'sklearn'] : ['scratch'],
  defaults: id === 'knn' ? { k: 5, distance: 'euclidean', weights: 'distance' } : id === 'naive_bayes' ? { var_smoothing: 1e-9 } : {},
  parameter_schema: id === 'knn' ? [
    { key: 'k', label: '近邻数 K', type: 'number', min: 1, max: 31, step: 2 },
    { key: 'distance', label: '距离', type: 'select', options: ['euclidean', 'manhattan'] },
    { key: 'weights', label: '投票', type: 'select', options: ['uniform', 'distance'] },
  ] : id === 'naive_bayes' ? [{ key: 'var_smoothing', label: '方差平滑', type: 'number', min: 1e-12, max: .01, step: 1e-9 }] : [],
}))

const records = new Map()

function demoResult(model, implementation, dataset) {
  const label = model === 'knn' ? 'KNN' : 'Naive Bayes'
  const row = aBenchmarkRows.find((item) => item.dataset.toLowerCase().replace(' ', '_') === dataset && item.model === label) || aBenchmarkRows.find((item) => item.model === label)
  const accuracy = row[implementation]
  const visualization = implementation === 'scratch' && model === 'knn' ? {
    type: 'knn_neighbors', class_names: ['setosa', 'versicolor', 'virginica'], query: { x: 0, y: 0 },
    neighbors: [1, 2, 3, 4, 5].map((rank) => ({ rank, x: Math.cos(rank) * rank, y: Math.sin(rank) * rank, label: 1, distance: rank * .13 })),
  } : implementation === 'scratch' ? {
    type: 'naive_bayes_stats', class_names: ['setosa', 'versicolor', 'virginica'], feature_names: ['sepal length', 'sepal width', 'petal length', 'petal width'],
    class_priors: [.333, .333, .334], means: [[5.0, 3.4, 1.5, .2], [5.9, 2.8, 4.3, 1.3], [6.6, 3.0, 5.5, 2.0]], variances: [[.12, .14, .03, .01], [.26, .10, .22, .04], [.40, .10, .30, .07]],
  } : {}
  return {
    id: `${model}:${implementation}`, model, task_type: 'classification', implementation, dataset,
    name: label, short_name: model === 'knn' ? 'KNN' : 'GNB', family: model === 'knn' ? '实例学习' : '概率模型',
    metrics: { accuracy, precision: accuracy, recall: accuracy, f1: accuracy }, training_ms: row[`${implementation}Train`], inference_ms: row[`${implementation}Predict`],
    params: model === 'knn' ? { k: 5, distance: 'euclidean', weights: 'distance' } : { var_smoothing: 1e-9 },
    confusion_matrix: [[10, 0, 0], [0, 10, 0], [0, 1, 9]], class_names: ['setosa', 'versicolor', 'virginica'], projection: [], training_curve: [], feature_importance: [], visualization,
    comparison: { accuracy_difference: 0, prediction_agreement: 1 },
  }
}

export const demoApi = {
  listDatasets: async () => datasets,
  listAlgorithms: async () => algorithms,
  listExperiments: async () => [...records.values()],
  getExperiment: async (id) => records.get(id),
  createExperiment: async (config) => {
    const id = `demo-${Date.now()}`
    const run_plan = config.model_ids.filter((model) => ['knn', 'naive_bayes'].includes(model)).flatMap((model) => config.implementations.map((implementation) => ({ model, implementation })))
    records.set(id, { id, status: 'queued', progress: 0, stage: '展示实验已创建', created_at: new Date().toISOString(), config, model_ids: config.model_ids, run_plan, results: [] })
    return { id, status: 'queued' }
  },
}

export function subscribeDemo(id, onRecord) {
  const record = records.get(id)
  const timers = []
  timers.push(window.setTimeout(() => { Object.assign(record, { status: 'running', progress: 35, stage: '载入固定基线' }); onRecord({ ...record }) }, 250))
  timers.push(window.setTimeout(() => {
    const results = record.run_plan.map((item) => demoResult(item.model, item.implementation, record.config.dataset_id))
    Object.assign(record, { status: 'completed', progress: 100, stage: '展示评估完成', results, best_model: results[0]?.id, best_accuracy: Math.max(...results.map((item) => item.metrics.accuracy)), dataset: datasets.find((item) => item.id === record.config.dataset_id) })
    onRecord({ ...record })
  }, 700))
  return () => timers.forEach(window.clearTimeout)
}
