export const aBenchmarkRows = [
  { dataset: 'Iris', model: 'KNN', scratch: 0.966667, sklearn: 0.966667, agreement: 1, scratchTrain: 0.019, scratchPredict: 0.459, sklearnTrain: 0.378, sklearnPredict: 0.394 },
  { dataset: 'Iris', model: 'Naive Bayes', scratch: 0.966667, sklearn: 0.966667, agreement: 1, scratchTrain: 0.088, scratchPredict: 0.031, sklearnTrain: 0.497, sklearnPredict: 0.111 },
  { dataset: 'Wine', model: 'KNN', scratch: 0.972222, sklearn: 0.972222, agreement: 1, scratchTrain: 0.012, scratchPredict: 0.711, sklearnTrain: 0.440, sklearnPredict: 0.448 },
  { dataset: 'Wine', model: 'Naive Bayes', scratch: 0.972222, sklearn: 0.972222, agreement: 1, scratchTrain: 0.108, scratchPredict: 0.042, sklearnTrain: 0.552, sklearnPredict: 0.138 },
  { dataset: 'Breast Cancer', model: 'KNN', scratch: 0.956140, sklearn: 0.956140, agreement: 1, scratchTrain: 0.029, scratchPredict: 7.324, sklearnTrain: 0.399, sklearnPredict: 11.776 },
  { dataset: 'Breast Cancer', model: 'Naive Bayes', scratch: 0.929825, sklearn: 0.929825, agreement: 1, scratchTrain: 0.179, scratchPredict: 0.056, sklearnTrain: 0.682, sklearnPredict: 0.163 },
]

export const knnParameterSeries = [
  { dataset: 'Iris', values: [{ k: 1, accuracy: .966667 }, { k: 3, accuracy: .966667 }, { k: 5, accuracy: .966667 }, { k: 7, accuracy: 1 }, { k: 9, accuracy: 1 }, { k: 11, accuracy: 1 }, { k: 15, accuracy: 1 }] },
  { dataset: 'Wine', values: [{ k: 1, accuracy: .972222 }, { k: 3, accuracy: .972222 }, { k: 5, accuracy: .972222 }, { k: 7, accuracy: 1 }, { k: 9, accuracy: 1 }, { k: 11, accuracy: 1 }, { k: 15, accuracy: 1 }] },
  { dataset: 'Breast Cancer', values: [{ k: 1, accuracy: .938596 }, { k: 3, accuracy: .982456 }, { k: 5, accuracy: .956140 }, { k: 7, accuracy: .973684 }, { k: 9, accuracy: .973684 }, { k: 11, accuracy: .973684 }, { k: 15, accuracy: .973684 }] },
]
