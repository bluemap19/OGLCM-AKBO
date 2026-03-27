"""
结果可视化模块
Visualization for OGLCM-AKBO Clustering Results

功能:
    1. 深度剖面图 (显示聚类标签随深度的分布)
    2. 特征分布箱线图 (显示各聚类中特征的分布)
    3. PCA 降维散点图 (二维投影可视化)
    4. 聚类中心雷达图 (显示各聚类中心特征)
    5. 特征相关性热图 (显示特征间相关性)
    6. 贝叶斯优化历史图 (显示优化过程)

作者：Cuka
日期：2026-03-17
版本：2.0 (支持特征选择后的可视化)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ClusteringVisualizer:
    """
    聚类结果可视化器
    
    属性:
        depth: ndarray, 深度数据 (N,)
        features: ndarray, 特征数据 (N×D)
        labels: ndarray, 聚类标签 (N,)
        feature_names: list, 特征名称列表
        n_clusters: int, 聚类数量
        colors: ndarray, 聚类颜色映射
    """
    
    def __init__(self, depth, features, labels, feature_names=None):
        """
        初始化可视化器
        
        参数:
            depth: ndarray, 深度数据 (N,)
            features: ndarray, 特征数据 (N×D)
            labels: ndarray, 聚类标签 (N,)
            feature_names: list, 特征名称列表 (可选)
        """
        self.depth = depth
        self.features = features
        self.labels = labels
        self.feature_names = feature_names if feature_names else [
            f'Feature {i}' for i in range(features.shape[1])
        ]
        self.n_clusters = len(np.unique(labels))
        
        # 使用 tab10  colormap (最多 10 个聚类)
        self.colors = plt.cm.tab10(np.linspace(0, 1, self.n_clusters))
    
    # def plot_depth_profile(self, save_path='results/figures/depth_profile.png'):
    #     """
    #     绘制深度剖面图（已删除）
    #     """
    #     pass
    
    def plot_feature_distribution(self, save_path='results/figures/feature_distribution.png'):
        """
        绘制特征分布箱线图（优化版）
        
        显示:
            - 每个特征在不同聚类中的分布 (箱线图)
            - 优化：更大的图、更清晰的标签、更好的颜色
        """
        print("\n[可视化] 绘制特征分布箱线图...")
        
        df = pd.DataFrame(self.features, columns=self.feature_names)
        df['Cluster'] = self.labels
        
        # 只绘制选中的 4 个特征
        n_features = len(self.feature_names)
        
        # 创建更大的图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # 使用更专业的配色方案
        palette = sns.color_palette("husl", self.n_clusters)
        
        for idx in range(n_features):
            ax = axes[idx]
            feat_name = self.feature_names[idx]
            
            # 准备数据
            data_to_plot = []
            cluster_labels = []
            for i in range(self.n_clusters):
                cluster_data = df[df['Cluster'] == i][feat_name].values
                data_to_plot.append(cluster_data)
                cluster_labels.append(f'Cluster {i}\n(n={len(cluster_data)})')
            
            # 绘制箱线图（使用更现代的样式）
            bp = ax.boxplot(
                data_to_plot, 
                labels=cluster_labels,
                patch_artist=True,
                widths=0.6,
                showfliers=False,  # 不显示异常值
                medianprops=dict(color='black', linewidth=2),  # 中位数线加粗
                boxprops=dict(linewidth=1.5),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5)
            )
            
            # 设置颜色（使用专业配色）
            for patch, color in zip(bp['boxes'], palette):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # 添加网格
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax.set_ylabel('Normalized Value', fontsize=11)
            ax.set_title(feat_name, fontsize=12, fontweight='bold', pad=10)
            ax.tick_params(axis='x', labelsize=9)
        
        # 隐藏多余的子图
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Feature Distribution by Cluster (Box Plot)', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"      已保存：{save_path}")
        plt.close()
    
    def plot_pca_scatter(self, save_path='results/figures/pca_scatter.png'):
        """
        绘制 PCA 降维散点图
        
        将高维特征投影到 2D 平面，显示聚类分离情况
        """
        print("\n[可视化] 绘制 PCA 降维散点图...")
        
        # PCA 降维
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(self.features)
        
        # 解释方差
        explained_var = pca.explained_variance_ratio_
        print(f"      PCA 解释方差：PC1={explained_var[0]*100:.1f}%, PC2={explained_var[1]*100:.1f}%")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 按聚类绘制散点
        for i in range(self.n_clusters):
            mask = self.labels == i
            ax.scatter(
                features_pca[mask, 0], 
                features_pca[mask, 1],
                c=[self.colors[i]], 
                label=f'Cluster {i}',
                alpha=0.6, 
                s=30, 
                edgecolors='k', 
                linewidth=0.5
            )
        
        ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}%)', fontsize=12)
        ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}%)', fontsize=12)
        ax.set_title('PCA Projection of Clusters', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"      已保存：{save_path}")
        plt.close()
    
    def plot_cluster_centers(self, save_path='results/figures/cluster_centers.png'):
        """
        绘制聚类中心雷达图
        
        显示每个聚类中心在各特征上的取值 (归一化后)
        """
        print("\n[可视化] 绘制聚类中心雷达图...")
        
        # 计算聚类中心
        centers = np.zeros((self.n_clusters, self.features.shape[1]))
        for i in range(self.n_clusters):
            centers[i] = self.features[self.labels == i].mean(axis=0)
        
        # 归一化到 [0, 1]
        centers_norm = (centers - centers.min()) / (centers.max() - centers.min() + 1e-10)
        
        # 选择部分特征 (最多 8 个)
        n_features = min(8, len(self.feature_names))
        feature_indices = np.linspace(
            0, 
            len(self.feature_names)-1, 
            n_features, 
            dtype=int
        )
        features_selected = [self.feature_names[i] for i in feature_indices]
        centers_selected = centers_norm[:, feature_indices]
        
        # 计算雷达图角度
        angles = np.linspace(0, 2 * np.pi, len(features_selected), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
        
        # 绘制每个聚类的雷达图
        for i in range(self.n_clusters):
            values = centers_selected[i].tolist()
            values += values[:1]  # 闭合
            ax.plot(
                angles, 
                values, 
                'o-', 
                linewidth=2, 
                label=f'Cluster {i}', 
                color=self.colors[i]
            )
            ax.fill(angles, values, alpha=0.15, color=self.colors[i])
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f[:15] for f in features_selected], fontsize=10)
        ax.set_title('Cluster Centers (Radar Chart)', fontsize=14, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"      已保存：{save_path}")
        plt.close()
    
    # def plot_correlation_heatmap(self, save_path='results/figures/correlation_heatmap.png'):
    #     """
    #     绘制特征相关性热图（已删除）
    #     """
    #     pass
    
    def plot_optimization_history(self, history, save_path='results/figures/optimization_history.png'):
        """
        绘制贝叶斯优化历史
        
        显示:
            - 左图：UIndex 随迭代变化
            - 右图：K 值选择历史
        """
        print("\n[可视化] 绘制优化历史...")
        
        if not history:
            print("      无优化历史记录")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 提取数据（兼容大小写键名）
        iteration_nums = list(range(1, len(history) + 1))
        uindex_values = [h.get('uindex', h.get('UIndex', 0)) for h in history]
        k_values = [h['K'] for h in history]
        
        # ---------------------------------------------------------------------
        # 左图：UIndex 随迭代变化
        # ---------------------------------------------------------------------
        ax = axes[0]
        ax.plot(
            iteration_nums, 
            uindex_values, 
            'bo-', 
            linewidth=2, 
            markersize=8
        )
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('UIndex', fontsize=12)
        ax.set_title('Bayesian Optimization History', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # ---------------------------------------------------------------------
        # 右图：K 值选择历史
        # ---------------------------------------------------------------------
        ax = axes[1]
        scatter = ax.scatter(
            iteration_nums, 
            k_values, 
            c=uindex_values, 
            cmap='viridis', 
            s=100, 
            alpha=0.6, 
            edgecolors='k'
        )
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('K Value', fontsize=12)
        ax.set_title('K Value Selection History', fontsize=14)
        plt.colorbar(scatter, label='UIndex')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"      已保存：{save_path}")
        plt.close()
    
    def plot_all(self, history=None):
        """
        绘制所有可视化图
        
        参数:
            history: list, 贝叶斯优化历史记录 (可选)
        """
        # self.plot_depth_profile()  # 已删除
        self.plot_feature_distribution()
        self.plot_pca_scatter()
        self.plot_cluster_centers()
        # self.plot_correlation_heatmap()  # 已删除
        
        if history:
            self.plot_optimization_history(history)
        
        print("\n[OK] 所有可视化图已生成!")


def visualize_results(depth, features, labels, feature_names=None, history=None):
    """
    可视化结果的便捷函数
    
    参数:
        depth: ndarray, 深度数据
        features: ndarray, 特征数据
        labels: ndarray, 聚类标签
        feature_names: list, 特征名称列表
        history: list, 优化历史记录
    
    返回:
        visualizer: ClusteringVisualizer, 可视化器对象
    """
    visualizer = ClusteringVisualizer(depth, features, labels, feature_names)
    visualizer.plot_all(history)
    
    return visualizer


if __name__ == "__main__":
    # 测试
    from data_loader import load_and_preprocess
    from akbo_clustering import akbo_clustering
    
    file_path = r"C:\Users\Maple\.openclaw\workspace\OGLCM-AKBO\TZ1H_texture_logging.csv"
    depth, features, preprocessor, report = load_and_preprocess(file_path)
    
    labels, clusterer = akbo_clustering(
        features, 
        k_range=(2, 6), 
        n_init=3, 
        max_iter=5
    )
    
    visualize_results(
        depth, 
        features, 
        labels, 
        preprocessor.feature_columns, 
        clusterer.optimization_history
    )
