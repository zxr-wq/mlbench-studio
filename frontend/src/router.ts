import { createRouter, createWebHashHistory } from 'vue-router'
import OverviewView from './views/OverviewView.vue'
import ExperimentView from './views/ExperimentView.vue'
import RunsView from './views/RunsView.vue'
import RunDetailView from './views/RunDetailView.vue'
import AlgorithmsView from './views/AlgorithmsView.vue'
import DatasetsView from './views/DatasetsView.vue'

const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: '/', name: 'overview', component: OverviewView, meta: { title: '概览', section: 'Lab 01' } },
    { path: '/experiments/new', name: 'experiment-new', component: ExperimentView, meta: { title: '创建实验', section: 'Builder' } },
    { path: '/runs', name: 'runs', component: RunsView, meta: { title: '运行记录', section: 'Experiments' } },
    { path: '/runs/:id', name: 'run-detail', component: RunDetailView, meta: { title: '实验详情', section: 'Experiments' } },
    { path: '/algorithms', name: 'algorithms', component: AlgorithmsView, meta: { title: '算法库', section: 'Registry' } },
    { path: '/datasets', name: 'datasets', component: DatasetsView, meta: { title: '数据集', section: 'Registry' } },
  ],
  scrollBehavior: () => ({ top: 0 }),
})

export default router
