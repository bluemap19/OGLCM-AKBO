"""
OGLCM-AKBO 无监督聚类算法主程序
Main Program for OGLCM-AKBO Unsupervised Clustering

基于定向灰度共生矩阵 (OGLCM) 与贝叶斯优化自适配 K-means (AKBO)
的页岩微相自动表征方法

输入：
    - TZ1H_texture_logging.csv: 纹理特征数据 (28 个 OGLCM 特征)

输出：
    - results/clustering_results.csv: 聚类结果
    - results/clustering_metrics.json: 评估指标
    - results/figures/*.png: 可视化图
    - results/test_report.md: 测试报告

作者：Doctor (Fuhao Zhang) & Cuka
日期：2026-03-17
版本：2.0 (添加特征选择，使用 4 个重要特征)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import load_and_preprocess_manual
from src.akbo_clustering import AKBOClusterer
from src.visualization import ClusteringVisualizer


def main():
    """
    主函数
    
    流程:
        1. 数据加载与预处理
        2. 特征选择 (使用 4 个重要纹理特征)
        3. AKBO 聚类优化
        4. 结果可视化
        5. 保存结果
    """
    
    print("="*80)
    print("OGLCM-AKBO 无监督聚类算法")
    print("基于定向灰度共生矩阵与贝叶斯优化自适配 K-means")
    print("="*80)
    print(f"开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # ==================== Step 1: 数据加载与预处理 ====================
    print("="*80)
    print("Step 1: 数据加载与预处理")
    print("="*80)
    
    data_file = r"C:\Users\Maple\.openclaw\workspace\OGLCM-AKBO\TZ1H_texture_logging.csv"
    
    # 预设的 4 个最优特征
    selected_features = [
        'CON_SUB_DYNA',
        'DIS_SUB_DYNA',
        'HOM_SUB_DYNA',
        'ENG_SUB_DYNA'
    ]
    
    # 加载数据并预处理 (自动选择指定特征)
    features, preprocessor, quality_report = load_and_preprocess_manual(
        data_file,
        feature_columns=selected_features
    )
    
    # 读取深度数据
    import pandas as pd
    df = pd.read_csv(data_file)
    depth = df['DEPTH'].values
    
    print(f"\n[OK] 数据加载完成")
    print(f"  - 样本数：{len(depth)}")
    print(f"  - 特征数：{features.shape[1]}")
    print(f"  - 深度范围：{depth.min():.2f} - {depth.max():.2f} m")
    
    # ==================== Step 2: 确认使用的特征 ====================
    print("\n" + "="*80)
    print("Step 2: 确认使用的特征")
    print("="*80)
    
    # 特征已在数据预处理阶段选择完成
    selected_feature_names = preprocessor.selected_columns
    selected_indices = list(range(len(selected_feature_names)))
    features_selected = features  # 已经是选择后的特征
    
    print(f"\n使用预设的 {len(selected_indices)} 个最优特征:")
    for i, name in enumerate(selected_feature_names):
        print(f"  {i+1}. {name}")
    
    # ==================== Step 3: AKBO 聚类优化 ====================
    print("\n" + "="*80)
    print("Step 3: AKBO 聚类优化")
    print("="*80)
    
    # 创建聚类器
    clusterer = AKBOClusterer(
        k_range=(5, 20),    # K 值搜索范围
        n_init=8,           # 初始采样点数
        max_iter=30,        # 最大迭代次数
        n_patience=8,       # 收敛等待次数
        tol=1e-4,           # 收敛阈值
        random_state=42     # 随机种子
    )
    
    # 执行优化 (使用指定的特征索引)
    optimal_k = clusterer.optimize(
        features,
        feature_names=selected_feature_names,
        selected_indices=selected_indices,
    )
    labels = clusterer.fit(features_selected)
    
    # ==================== Step 4: 结果可视化 ====================
    print("\n" + "="*80)
    print("Step 4: 结果可视化")
    print("="*80)
    
    visualizer = ClusteringVisualizer(
        depth=depth,
        features=features_selected,  # 使用选中的特征进行可视化
        labels=labels,
        feature_names=selected_feature_names
    )

    # 绘制所有图表
    visualizer.plot_all(history=clusterer.optimization_history)
    
    # ==================== Step 5: 保存结果 ====================
    print("\n" + "="*80)
    print("Step 5: 保存结果")
    print("="*80)
    
    # 1. 保存聚类结果
    results_df = pd.DataFrame({
        'DEPTH': depth,
        'CLUSTER_LABEL': labels
    })
    
    # 添加聚类概率 (基于 GMM 后验概率)
    probs = clusterer.get_cluster_probs(features_selected)
    for i in range(probs.shape[1]):
        results_df[f'CLUSTER_{i}_PROB'] = probs[:, i]
    
    results_file = 'results/clustering_results.csv'
    results_df.to_csv(results_file, index=False)
    print(f"[OK] 聚类结果已保存：{results_file}")
    
    # 2. 生成测试报告（包含详细迭代历史）
    generate_test_report(
        optimal_k=optimal_k,
        n_samples=len(depth),
        n_features=len(selected_indices),
        selected_features=selected_feature_names,
        depth_range=[float(depth.min()), float(depth.max())],
        best_metrics=clusterer.best_metrics,
        cluster_distribution={f'cluster_{i}': int(np.sum(labels == i)) for i in range(optimal_k)},
        optimization_history=clusterer.optimization_history,
        results_file=results_file
    )
    
    # ==================== 完成 ====================
    print("\n" + "="*80)
    print("[OK] 所有任务完成!")
    print("="*80)
    print(f"结束时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n输出文件:")
    print(f"  1. 聚类结果：{results_file}")
    print(f"  2. 可视化图：results/figures/")
    print(f"  3. 测试报告：results/test_report.md (包含详细迭代历史)")
    
    return results_df, clusterer


def generate_test_report(optimal_k, n_samples, n_features, selected_features, 
                         depth_range, best_metrics, cluster_distribution, 
                         optimization_history, results_file):
    """
    生成测试报告（包含详细迭代历史）
    
    参数:
        optimal_k: int, 最优 K 值
        n_samples: int, 样本数
        n_features: int, 特征数
        selected_features: list, 特征名称
        depth_range: list, 深度范围
        best_metrics: dict, 最优指标
        cluster_distribution: dict, 聚类分布
        optimization_history: list, 优化历史
        results_file: str, 结果文件路径
    """
    
    # 质量评价
    sil = best_metrics['si']
    if sil > 0.7:
        quality_rating = "优秀 - 聚类结构清晰，分离度好"
    elif sil > 0.5:
        quality_rating = "良好 - 聚类结构合理，可以接受"
    elif sil > 0.25:
        quality_rating = "一般 - 聚类结构较弱，建议调整参数"
    else:
        quality_rating = "较差 - 聚类结构不明显，需要重新考虑"
    
    from datetime import datetime
    timestamp = datetime.now().isoformat()
    
    report = f"""# OGLCM-AKBO 聚类测试报告

**生成时间：** {timestamp}

---

## 📊 数据概况

| 项目 | 值 |
|------|-----|
| **样本数量** | {n_samples} |
| **特征数量** | {n_features} |
| **深度范围** | {depth_range[0]:.2f} - {depth_range[1]:.2f} m |
| **最优聚类数 (K)** | {optimal_k} |

---

## 🔬 特征选择

**选中的重要特征 (共{len(selected_features)}个):**

| # | 特征名 | 地质意义 |
|---|--------|----------|
| 1 | {selected_features[0]} | 对比度_子区域_动态 - 反映局部纹理变化强度 |
| 2 | {selected_features[1]} | 差异性_子区域_动态 - 反映子区域灰度差异 |
| 3 | {selected_features[2]} | 同质性_子区域_动态 - 反映子区域纹理均匀度 |
| 4 | {selected_features[3]} | 熵_子区域_动态 - 反映纹理复杂度 |

**特征选择依据:** 基于随机森林特征重要性分析（已在数据预处理阶段完成）

---

## 🎯 聚类质量评估

| 指标 | 值 | 说明 | 评价 |
|------|-----|------|------|
| **UIndex** | {best_metrics['uindex']:.4f} | 综合指标 (越大越好) | {'✅ 优秀' if best_metrics['uindex'] > 0.5 else '⚠️ 需改进'} |
| **轮廓系数 (SI)** | {best_metrics['si']:.4f} | >0.5 表示聚类合理 | {'✅ 良好' if best_metrics['si'] > 0.5 else '⚠️ 一般'} |
| **DBI** | {best_metrics['dbi']:.4f} | <1.0 为好 | {'✅ 良好' if best_metrics['dbi'] < 1.0 else '⚠️ 需改进'} |
| **DVI** | {best_metrics['dvi']:.4f} | 越大越好 | {'✅ 良好' if best_metrics['dvi'] > 0.5 else '⚠️ 较低'} |

### 聚类质量评价:

**{quality_rating}**

---

## 📈 聚类分布

| 聚类标签 | 样本数 | 占比 (%) | 地质解释推测 |
|---------|--------|----------|-------------|
"""
    
    for i in range(optimal_k):
        count = cluster_distribution[f'cluster_{i}']
        percentage = count / n_samples * 100
        report += f"| Cluster {i} | {count} | {percentage:.1f}% | 待解释 |\n"
    
    report += f"""
---

## 🔄 贝叶斯优化历史（详细迭代记录）

**总迭代次数：** {len(optimization_history)} 次（包括初始采样和贝叶斯优化）

### 完整迭代历史

| 迭代 | K 值 | UIndex | SI | DBI | DVI | 改进 |
|------|-----|--------|----|----|----|------|
"""
    
    for record in optimization_history:
        iteration = record['iteration']
        k = record['K']
        uindex = record['UIndex']
        si = record['SI']
        dbi = record['DBI']
        dvi = record['DVI']
        improved = '✅' if record['improved'] else '❌'
        report += f"| {iteration} | {k} | {uindex:.4f} | {si:.4f} | {dbi:.4f} | {dvi:.4f} | {improved} |\n"
    
    report += f"""
---

## 📁 输出文件

1. **聚类结果:** {results_file}
   - DEPTH: 深度
   - CLUSTER_LABEL: 聚类标签
   - CLUSTER_X_PROB: 属于各聚类的概率 (基于 GMM 后验概率)

2. **可视化图:** results/figures/
   - depth_profile.png: 深度剖面图
   - feature_distribution.png: 特征分布箱线图
   - pca_scatter.png: PCA 降维散点图
   - cluster_centers.png: 聚类中心雷达图
   - correlation_heatmap.png: 特征相关性热图

3. **测试报告:** results/test_report.md (本文档)

---

## ⚙️ 算法参数

| 参数 | 值 |
|------|-----|
| K 值搜索范围 | [5, 20] |
| 初始采样点数 | 5 |
| 最大迭代次数 | 30 |
| 收敛等待次数 | 5 |
| 收敛阈值 | 1e-4 |

*报告由 OGLCM-AKBO 算法自动生成*
"""
    
    # 保存报告
    report_file = 'results/test_report.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"[OK] 测试报告已保存：{report_file}")


if __name__ == "__main__":
    # 确保结果目录存在
    os.makedirs('results/figures', exist_ok=True)
    
    # 运行主程序
    results, clusterer = main()
