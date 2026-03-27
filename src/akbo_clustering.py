"""
AKBO 核心聚类算法模块
Auto-Kmeans with Bayesian Optimization Clustering

功能:
    1. 实现复合聚类质量指标 UIndex (SI + DBI + DVI)
    2. 实现高斯过程代理模型
    3. 实现贝叶斯优化循环 (使用正确的 EI 采集函数)
    4. 实现 K-means++ 初始化
    5. 实现基于 GMM 的聚类概率计算

注意:
    特征选择已在数据预处理阶段完成，使用预设的 4 个最优特征：
    ['CON_SUB_DYNA', 'DIS_SUB_DYNA', 'HOM_SUB_DYNA', 'ENG_SUB_DYNA']
    这些特征是基于前期随机森林特征重要性分析确定的

作者：Cuka & Doctor (Fuhao Zhang)
日期：2026-03-17
版本：2.1 (移除随机森林特征选择，使用预设特征)
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist, pdist
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# 聚类质量指标计算函数
# ============================================================================

def compute_dunn_index(X, labels, use_approximation=True):
    """
    计算 Dunn Index (DVI) - 衡量聚类的紧凑性和分离度
    
    Dunn Index = min(簇间距离) / max(簇内直径)
    值越大表示聚类质量越好
    
    参数:
        X: ndarray, 特征数据 (N×D)
            N: 样本数，D: 特征维度
        labels: ndarray, 聚类标签 (N,)
            每个样本的聚类标签 (0, 1, ..., K-1)
        use_approximation: bool, 默认 True
            是否使用近似算法 (到质心最大距离×2)
            True: O(N) 复杂度，快速
            False: O(N²) 复杂度，精确
    
    返回:
        dvi: float, Dunn Index 值
            - dvi > 1.0: 优秀
            - dvi > 0.5: 良好
            - dvi < 0.5: 可接受
    
    注意:
        - 当簇内直径为 0 时返回无穷大
        - 当聚类数 < 2 时返回 0.0
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # 边界情况：聚类数少于 2
    if n_clusters < 2:
        return 0.0
    
    # -------------------------------------------------------------------------
    # 步骤 1: 计算簇内最大直径
    # -------------------------------------------------------------------------
    max_diameter = 0.0
    centers = []
    
    for label in unique_labels:
        cluster_points = X[labels == label]
        
        # 计算簇质心
        center = cluster_points.mean(axis=0)
        centers.append(center)
        
        # 计算簇直径
        if len(cluster_points) > 1:
            if use_approximation:
                # 近似算法：直径 ≈ 2 × 最大半径 (到质心的距离)
                # 复杂度：O(N)，适合大数据集
                distances = np.linalg.norm(cluster_points - center, axis=1)
                diameter = 2 * np.max(distances)
            else:
                # 精确算法：计算所有点对距离的最大值
                # 复杂度：O(N²)，适合小数据集
                distances = pdist(cluster_points)
                diameter = np.max(distances) if len(distances) > 0 else 0.0
            
            max_diameter = max(max_diameter, diameter)
    
    # 边界情况：所有簇直径为 0
    if max_diameter == 0:
        return float('inf')
    
    # -------------------------------------------------------------------------
    # 步骤 2: 计算簇间最小距离 (使用质心距离)
    # -------------------------------------------------------------------------
    centers = np.array(centers)
    center_distances = cdist(centers, centers, metric='euclidean')
    
    # 忽略对角线 (自身到自身的距离)
    np.fill_diagonal(center_distances, np.inf)
    min_distance = np.min(center_distances)
    
    # -------------------------------------------------------------------------
    # 步骤 3: 计算 Dunn Index
    # -------------------------------------------------------------------------
    dvi = min_distance / max_diameter
    
    return dvi


def compute_uindex(X, labels):
    """
    计算复合聚类质量指标 UIndex
    
    UIndex 综合三个经典指标，基于 Doctor 提出的公式：
        UIndex(K) = 1 / (0.1/SI(K) + DBI(K)/1.0 + 0.01/DVI(K))
    
    其中:
        - SI (Silhouette Index): 轮廓系数，衡量样本聚类质量 [-1, 1]，越大越好
        - DBI (Davies-Bouldin Index): 簇间分离度 [0, ∞)，越小越好
        - DVI (Dunn Index): 簇紧凑性 [0, ∞)，越大越好
    
    公式设计原理:
        - SI 越大 → 0.1/SI 越小 → 分母越小 → UIndex 越大 ✅
        - DBI 越小 → DBI/1.0 越小 → 分母越小 → UIndex 越大 ✅
        - DVI 越大 → 0.01/DVI 越小 → 分母越小 → UIndex 越大 ✅
    
    参数:
        X: ndarray, 特征数据 (N×D)
        labels: ndarray, 聚类标签 (N,)
    
    返回:
        uindex: float, 综合指标值 (越大越好)
        metrics: dict, 包含各分项指标
            {
                'si': float,      # 轮廓系数
                'dbi': float,     # Davies-Bouldin Index
                'dvi': float,     # Dunn Index
                'uindex': float   # 综合指标
            }
    
    质量评价标准:
        - UIndex > 0.5: 优秀
        - UIndex > 0.2: 良好
        - UIndex < 0.2: 需改进
    """
    n_samples = len(labels)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # 边界情况：单样本或单聚类
    if n_samples < 2 or n_clusters < 2:
        return 0.0, {'si': 0.0, 'dbi': float('inf'), 'dvi': 0.0, 'uindex': 0.0}
    
    # -------------------------------------------------------------------------
    # 步骤 1: 计算轮廓系数 SI (Silhouette Index)
    # -------------------------------------------------------------------------
    si = silhouette_score(X, labels, metric='euclidean')
    
    # -------------------------------------------------------------------------
    # 步骤 2: 计算 Davies-Bouldin Index (DBI)
    # -------------------------------------------------------------------------
    dbi = davies_bouldin_score(X, labels)
    
    # -------------------------------------------------------------------------
    # 步骤 3: 计算 Dunn Index (DVI)
    # -------------------------------------------------------------------------
    dvi = compute_dunn_index(X, labels, use_approximation=True)
    
    # 处理无穷大和零值情况
    if np.isinf(dvi) or dvi == 0:
        dvi = 10.0  # 给一个较大的值
    if np.isinf(dbi):
        dbi = 0.0001  # 避免除以 0
    if si <= 0:
        si = 0.0001  # 避免除以 0
    
    # -------------------------------------------------------------------------
    # 步骤 4: 计算 UIndex (使用 Doctor 提出的公式)
    # -------------------------------------------------------------------------
    # UIndex(K) = 1 / (0.1/SI(K) + DBI(K)/1.0 + 0.01/DVI(K))
    uindex = 1.0 / (0.1/si + dbi/1.0 + 0.01/dvi)
    
    metrics = {
        'uindex': float(uindex),
        'si': float(si),
        'dbi': float(dbi),
        'dvi': float(dvi)
    }
    
    return uindex, metrics


# ============================================================================
# AKBOClusterer 类 - 贝叶斯优化自适配 K-means 聚类器
# ============================================================================

class AKBOClusterer:
    """
    AKBO 聚类器 (Auto-Kmeans with Bayesian Optimization)
    
    核心思想:
        将 K 值选择公式化为黑盒优化问题，使用贝叶斯优化自动寻找最优 K 值
    
    优化目标:
        max K∈[Kmin,Kmax] UIndex(K)
    
    属性:
        k_range: tuple, K 值搜索范围 (K_min, K_max)
        n_init: int, 初始采样点数
        max_iter: int, 贝叶斯优化最大迭代次数
        n_patience: int, 收敛等待次数 (连续无改进次数)
        tol: float, 收敛阈值
        random_state: int, 随机种子
        best_k: int, 最优 K 值
        best_labels: ndarray, 最优聚类标签
        best_model: KMeans, 最优 K-means 模型
        best_metrics: dict, 最优聚类指标
        optimization_history: list, 优化历史记录
        selected_features: list, 选中的特征索引
        gmm_model: GaussianMixture, GMM 模型 (用于概率计算)
    
    注意:
        特征选择已在数据预处理阶段完成，使用预设的 4 个最优特征：
        ['CON_SUB_DYNA', 'DIS_SUB_DYNA', 'HOM_SUB_DYNA', 'ENG_SUB_DYNA']
        这些特征是基于前期随机森林特征重要性分析确定的
    
    示例:
        >>> clusterer = AKBOClusterer(k_range=(2, 10), max_iter=30)
        >>> optimal_k = clusterer.optimize(X, feature_names, selected_indices)
        >>> labels = clusterer.fit(X)
    """
    
    def __init__(self, k_range=(2, 10), n_init=5, max_iter=30, 
                 n_patience=5, tol=1e-4, random_state=42):
        """
        初始化 AKBO 聚类器
        
        参数:
            k_range: tuple, 默认 (2, 10)
                K 值搜索范围 [K_min, K_max)
            n_init: int, 默认 5
                初始采样点数 (随机选择的 K 值数量)
            max_iter: int, 默认 30
                贝叶斯优化最大迭代次数
            n_patience: int, 默认 5
                收敛等待次数 (连续无改进则提前终止)
            tol: float, 默认 1e-4
                收敛阈值 (UIndex 改进小于此值视为无改进)
            random_state: int, 默认 42
                随机种子 (保证结果可复现)
        """
        self.k_range = k_range
        self.n_init = n_init
        self.max_iter = max_iter
        self.n_patience = n_patience
        self.tol = tol
        self.random_state = random_state
        
        # 优化结果
        self.best_k = None
        self.best_labels = None
        self.best_model = None
        self.best_metrics = None
        self.optimization_history = []  # 记录每次迭代的详细信息
        
        # 特征选择 (在数据预处理阶段完成)
        self.selected_features = None
        
        # GMM 模型 (用于概率计算)
        self.gmm_model = None
    
    def _expected_improvement(self, X, gp, best_f):
        """
        计算 Expected Improvement (EI) 采集函数
        
        EI 公式:
            EI(x) = E[max(0, f(x) - f_best)]
                  = (μ - f_best) × Φ(Z) + σ × φ(Z)
        
        其中:
            - μ: GP 预测均值
            - σ: GP 预测标准差
            - Z = (μ - f_best) / σ
            - Φ: 标准正态分布 CDF
            - φ: 标准正态分布 PDF
        
        参数:
            X: ndarray, 候选 K 值 (N×1)
            gp: GaussianProcessRegressor, 训练好的 GP 模型
            best_f: float, 当前最优 UIndex 值
        
        返回:
            ei: ndarray, 每个候选 K 值的 EI 值 (N,)
        """
        # 获取 GP 预测 (均值和标准差)
        mu, sigma = gp.predict(X.reshape(-1, 1), return_std=True)
        
        # 计算 EI
        with np.errstate(divide='ignore'):
            imp = mu - best_f  # 改进量
            Z = imp / sigma    # 标准化
            
            # EI = (μ - f_best) × Φ(Z) + σ × φ(Z)
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            
            # 处理标准差为 0 的情况
            ei[sigma == 0.0] = 0.0
        
        return ei
    
    def _initialize_sampling(self, X):
        """
        初始采样：随机选择 K 值并评估 UIndex
        
        策略:
            - 从 K 值范围内随机选择 n_init 个不同的 K 值
            - 对每个 K 执行 K-means++ 聚类
            - 计算 UIndex 作为评估指标
        
        参数:
            X: ndarray, 标准化后的特征数据 (N×D)
        
        返回:
            K_values: list, 初始 K 值列表
            uindex_values: list, 对应的 UIndex 值
            models: dict, K-means 模型字典 {K: model}
        """
        print(f"\n[AKBO] 初始采样 ({self.n_init} 个点)...")
        
        K_values = []
        uindex_values = []
        models = {}
        
        # 设置随机种子保证可复现
        np.random.seed(self.random_state)
        
        # 随机选择 K 值 (不重复)
        candidate_K = np.random.choice(
            range(self.k_range[0], self.k_range[1]), 
            size=min(self.n_init, self.k_range[1] - self.k_range[0]), 
            replace=False
        )
        
        for i, k in enumerate(candidate_K):
            print(f"      采样 {i+1}/{self.n_init}: K={k}", end=' ')
            
            # K-means++ 聚类
            # init='k-means++': 智能初始化质心，加速收敛
            # n_init=10: 运行 10 次，选择最优结果
            kmeans = KMeans(
                n_clusters=k,
                init='k-means++',
                n_init=10,
                random_state=self.random_state,
                max_iter=300
            )
            labels = kmeans.fit_predict(X)
            
            # 计算 UIndex
            uindex, metrics = compute_uindex(X, labels)
            
            K_values.append(k)
            uindex_values.append(uindex)
            models[k] = kmeans
            
            print(f"-> UIndex={uindex:.4f} (SI={metrics['si']:.3f})")
        
        return K_values, uindex_values, models
    
    def _bayesian_optimization(self, X, K_values, uindex_values, models):
        """
        贝叶斯优化迭代
        
        流程:
            1. 基于已有数据训练高斯过程 (GP) 代理模型
            2. 使用 EI 采集函数选择下一个候选 K 值
            3. 执行 K-means 聚类并计算 UIndex
            4. 更新 GP 模型和数据集
            5. 检查收敛条件
        
        参数:
            X: ndarray, 特征数据 (N×D)
            K_values: list, 已评估的 K 值列表
            uindex_values: list, 对应的 UIndex 值列表
            models: dict, 已训练的 K-means 模型字典
        
        返回:
            K_values: list, 更新后的 K 值列表
            uindex_values: list, 更新后的 UIndex 值列表
            models: dict, 更新后的模型字典
        """
        print(f"\n[AKBO] 贝叶斯优化迭代 (最多{self.max_iter}次)...")
        
        # 当前最优 UIndex
        best_uindex = max(uindex_values)
        patience_counter = 0
        
        # -------------------------------------------------------------------------
        # 初始化高斯过程代理模型
        # -------------------------------------------------------------------------
        # 核函数：RBF (径向基函数) + ConstantKernel
        # RBF: 捕捉 K 值之间的平滑关系
        # ConstantKernel: 缩放信号方差
        kernel = C(1.0) * RBF(length_scale=1.0)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            random_state=self.random_state
        )
        
        # -------------------------------------------------------------------------
        # 贝叶斯优化循环
        # -------------------------------------------------------------------------
        for iteration in range(self.max_iter):
            # 当前评估点数
            n_evaluated = len(K_values)
            
            # 准备训练数据 (K 值作为输入，UIndex 作为输出)
            X_gp = np.array(K_values).reshape(-1, 1)
            y_gp = np.array(uindex_values)
            
            # 训练 GP 模型
            gp.fit(X_gp, y_gp)
            
            # -------------------------------------------------------------------------
            # 使用 EI 采集函数选择下一个 K 值
            # -------------------------------------------------------------------------
            # 候选 K 值 (所有未评估的整数 K)
            candidate_K = np.array([
                k for k in range(self.k_range[0], self.k_range[1])
                if k not in K_values
            ])
            
            if len(candidate_K) == 0:
                print(f"      迭代 {iteration+1}/{self.max_iter}: 所有 K 值已评估")
                break
            
            # 计算每个候选 K 值的 EI
            ei_values = self._expected_improvement(candidate_K, gp, best_uindex)
            
            # 选择 EI 最大的 K 值
            next_idx = np.argmax(ei_values)
            next_k = candidate_K[next_idx]
            
            # -------------------------------------------------------------------------
            # 执行 K-means 聚类并计算 UIndex
            # -------------------------------------------------------------------------
            print(f"      迭代 {iteration+1}/{self.max_iter}: K={next_k}", end=' ')
            
            kmeans = KMeans(
                n_clusters=next_k,
                init='k-means++',
                n_init=10,
                random_state=self.random_state,
                max_iter=300
            )
            labels = kmeans.fit_predict(X)
            
            uindex, metrics = compute_uindex(X, labels)
            
            # -------------------------------------------------------------------------
            # 记录迭代历史（详细指标）
            # -------------------------------------------------------------------------
            iteration_record = {
                'iteration': iteration + 1,
                'K': next_k,
                'UIndex': uindex,
                'SI': metrics['si'],
                'DBI': metrics['dbi'],
                'DVI': metrics['dvi'],
                'improved': uindex > best_uindex
            }
            self.optimization_history.append(iteration_record)
            
            # -------------------------------------------------------------------------
            # 更新数据集
            # -------------------------------------------------------------------------
            K_values.append(next_k)
            uindex_values.append(uindex)
            models[next_k] = kmeans
            
            print(f"-> UIndex={uindex:.4f} (SI={metrics['si']:.3f}, DBI={metrics['dbi']:.3f}, DVI={metrics['dvi']:.3f})")
            
            # -------------------------------------------------------------------------
            # 检查是否改进
            # -------------------------------------------------------------------------
            if uindex > best_uindex:
                improvement = uindex - best_uindex
                best_uindex = uindex
                patience_counter = 0
                print(f"         [改进] +{improvement:.4f}")
            else:
                patience_counter += 1
                print(f"         [无改进] 等待{patience_counter}/{self.n_patience}")
            
            # -------------------------------------------------------------------------
            # 收敛检查
            # -------------------------------------------------------------------------
            if patience_counter >= self.n_patience:
                print(f"\n[AKBO] 达到收敛条件 (连续{self.n_patience}次无改进)")
                break
        
        return K_values, uindex_values, models
    
    def select_features(self, X, selected_indices, feature_names):
        """
        根据指定的特征索引选择特征
        
        注意：特征选择已在数据预处理阶段完成，此方法仅用于提取指定特征
        
        参数:
            X: ndarray, 原始特征数据 (N×D)
            selected_indices: list, 特征索引列表
            feature_names: list, 特征名称列表
        
        返回:
            X_selected: ndarray, 选择后的特征数据 (N×n_select)
            selected_names: list, 选中的特征名称
        """
        # 选择特征
        X_selected = X[:, selected_indices]
        selected_names = [feature_names[i] for i in selected_indices]
        
        self.selected_features = selected_indices
        
        print(f"\n[特征选择] 已选择 {len(selected_indices)} 个特征:")
        for i, idx in enumerate(selected_indices):
            print(f"  {i+1}. {feature_names[idx]}")
        
        return X_selected, selected_names
    
    def optimize(self, X, feature_names=None, selected_indices=None):
        """
        执行 AKBO 优化
        
        流程:
            1. 使用指定的特征进行聚类（特征选择已在数据预处理阶段完成）
            2. 初始采样
            3. 贝叶斯优化
            4. 选择最优 K 值
        
        参数:
            X: ndarray, 标准化后的特征数据 (N×D)
            feature_names: list, 特征名称列表
            selected_indices: list, 指定的特征索引列表
                             默认使用全部特征
        
        返回:
            best_k: int, 最优 K 值
        
        注意:
            特征选择已在数据预处理阶段完成，使用预设的 4 个最优特征：
            ['CON_SUB_DYNA', 'DIS_SUB_DYNA', 'HOM_SUB_DYNA', 'ENG_SUB_DYNA']
            这些特征是基于前期随机森林特征重要性分析确定的
        """
        print("="*60)
        print("AKBO 聚类优化")
        print("="*60)
        print(f"K 值搜索范围：[{self.k_range[0]}, {self.k_range[1]}]")
        print(f"初始采样点数：{self.n_init}")
        print(f"最大迭代次数：{self.max_iter}")
        
        # -------------------------------------------------------------------------
        # 步骤 1: 确定最终使用的特征
        # -------------------------------------------------------------------------
        if selected_indices is not None and feature_names is not None:
            # 使用指定的特征索引
            X_opt, selected_names = self.select_features(X, selected_indices, feature_names)
            print(f"\n使用指定的特征进行优化 ({len(selected_names)} 个特征)")
        else:
            # 使用全部特征
            X_opt = X
            selected_names = feature_names
            print("\n[特征处理] 使用全部特征进行优化。")
        
        # -------------------------------------------------------------------------
        # 步骤 2: 初始采样（记录历史）
        # -------------------------------------------------------------------------
        K_values, uindex_values, models = self._initialize_sampling(X_opt)
        
        # 记录初始采样到历史
        for i, (k, uindex) in enumerate(zip(K_values, uindex_values)):
            _, metrics = compute_uindex(X_opt, models[k].labels_)
            self.optimization_history.append({
                'iteration': f'initial_{i+1}',
                'K': k,
                'UIndex': uindex,
                'SI': metrics['si'],
                'DBI': metrics['dbi'],
                'DVI': metrics['dvi'],
                'improved': True
            })
        
        # -------------------------------------------------------------------------
        # 步骤 3: 贝叶斯优化
        # -------------------------------------------------------------------------
        K_values, uindex_values, models = self._bayesian_optimization(
            X_opt, K_values, uindex_values, models
        )
        
        # -------------------------------------------------------------------------
        # 步骤 4: 选择最优结果
        # -------------------------------------------------------------------------
        best_idx = np.argmax(uindex_values)
        self.best_k = K_values[best_idx]
        self.best_model = models[self.best_k]
        self.best_labels = self.best_model.labels_
        
        # 计算最优指标
        _, self.best_metrics = compute_uindex(X_opt, self.best_labels)
        
        # 训练 GMM 模型用于概率计算
        self.gmm_model = GaussianMixture(
            n_components=self.best_k,
            covariance_type='full',
            random_state=self.random_state,
            n_init=10
        )
        self.gmm_model.fit(X_opt)
        
        print("\n" + "="*60)
        print(f"[OK] AKBO 优化完成")
        print(f"最优 K 值：{self.best_k}")
        print(f"最优 UIndex: {self.best_metrics['uindex']:.4f}")
        print(f"  - 轮廓系数 (SI): {self.best_metrics['si']:.4f}")
        print(f"  - DBI: {self.best_metrics['dbi']:.4f}")
        print(f"  - DVI: {self.best_metrics['dvi']:.4f}")
        print("="*60)
        
        return self.best_k
    
    def fit(self, X):
        """
        执行聚类 (如果尚未优化，先优化)
        
        参数:
            X: ndarray, 特征数据
        
        返回:
            labels: ndarray, 聚类标签 (N,)
        """
        if self.best_model is None:
            print("[警告] 尚未执行 optimize()，现在执行自动优化...")
            self.optimize(X)
        
        return self.best_labels
    
    def predict(self, X):
        """
        预测新样本的聚类标签
        参数:
            X: ndarray, 新样本特征数据 (M×D)
        返回:
            labels: ndarray, 预测的聚类标签 (M,)
        """
        if self.best_model is None:
            raise ValueError("尚未执行优化，请先调用 optimize() 或 fit()")
        
        return self.best_model.predict(X)
    
    def get_cluster_probs(self, X):
        """
        获取聚类概率 (基于 GMM 后验概率)
        参数:
            X: ndarray, 特征数据 (N×D)
        返回:
            probs: ndarray, 聚类概率 (N×K)
                每行表示样本属于各聚类的概率
        注意:
            GMM 模型在 optimize() 结束后自动训练
        """
        if self.gmm_model is None:
            raise ValueError("GMM 模型未训练，请先调用 optimize()")
        
        return self.gmm_model.predict_proba(X)


# ============================================================================
# 便捷函数
# ============================================================================
def akbo_clustering(X, k_range=(2, 10), max_iter=30, random_state=42):
    """
    便捷函数：执行 AKBO 聚类
    
    参数:
        X: ndarray, 标准化后的特征数据 (N×D)
        k_range: tuple, K 值搜索范围
        max_iter: int, 最大迭代次数
        random_state: int, 随机种子
    
    返回:
        labels: ndarray, 聚类标签 (N,)
        optimal_k: int, 最优 K 值
        metrics: dict, 聚类质量指标
    """
    clusterer = AKBOClusterer(
        k_range=k_range,
        max_iter=max_iter,
        random_state=random_state
    )
    
    optimal_k = clusterer.optimize(X)
    labels = clusterer.fit(X)
    
    return labels, optimal_k, clusterer.best_metrics
