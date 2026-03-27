"""
手动特征选择的数据加载与预处理模块
Manual Feature Selection Data Loader and Preprocessing for OGLCM-AKBO

功能:
    1. 读取 CSV 纹理特征数据
    2. 根据用户指定的特征列名列表选择数据
    3. 数据质量检查 (缺失值、异常值)
    4. 特征标准化 (Z-score)

作者：基于原版修改
日期：2026-03-17
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')


class ManualFeaturePreprocessor:
    """
    手动特征选择预处理器

    属性:
        scaler: StandardScaler 对象，用于特征标准化
        imputer: SimpleImputer 对象，用于缺失值填充
        selected_columns: 用户指定的特征列名列表
    """

    def __init__(self, feature_columns):
        """
        初始化预处理器

        参数:
            feature_columns: list, 用户指定的特征列名列表
               例如: ['GLCM_Contrast', 'GLCM_Correlation', 'GLCM_Energy']
        """
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.selected_columns = feature_columns

    def load_and_select_data(self, file_path):
        """
        加载 CSV 数据并选择指定特征

        参数:
            file_path: CSV 文件路径

        返回:
            features: DataFrame, 只包含用户指定特征的数据

        异常:
            ValueError: 如果指定的特征列在数据中不存在
        """
        print(f"[1/4] 正在加载数据：{file_path}")
        df = pd.read_csv(file_path)
        print(f"      原始数据形状：{df.shape}")
        print(f"      原始数据列名：{df.columns.tolist()}")

        # 检查所有指定特征列是否存在于数据中
        missing_columns = [col for col in self.selected_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"以下指定特征在数据中不存在: {missing_columns}")

        # 选择用户指定的特征列
        features = df[self.selected_columns].copy()

        print(f"      选择的特征数量：{len(self.selected_columns)}")
        print(f"      选择的特征：{self.selected_columns}")
        print(f"      处理后的数据形状：{features.shape}")

        return features

    def check_data_quality(self, features):
        """
        数据质量检查（仅在用户指定的特征上进行）
        参数:
            features: DataFrame, 特征数据
        返回:
            report: dict, 质量检查报告
        """
        report = {
            'missing_values': int(features.isnull().sum().sum()),
            'zero_variance': [],
            'outliers': {},
            'selected_features': self.selected_columns,
            'feature_statistics': {}
        }

        # 检查零方差特征
        for col in features.columns:
            std_val = features[col].std()
            if std_val < 1e-10:
                report['zero_variance'].append(col)

            # 记录每个特征的基本统计信息
            report['feature_statistics'][col] = {
                'mean': float(features[col].mean()),
                'std': float(std_val),
                'min': float(features[col].min()),
                'max': float(features[col].max()),
                'missing': int(features[col].isnull().sum())
            }

        # 检查异常值 (3σ原则)
        total_outliers = 0
        for col in features.columns:
            mean = features[col].mean()
            std = features[col].std()
            if std > 1e-10:  # 避免除零
                outliers = ((features[col] < mean - 3*std) | (features[col] > mean + 3*std)).sum()
                if outliers > 0:
                    report['outliers'][col] = {
                        'count': int(outliers),
                        'percentage': float(outliers / len(features) * 100)
                    }
                    total_outliers += outliers

        print("[2/4] 数据质量检查（基于选定特征）")
        print(f"      缺失值总数：{report['missing_values']}")
        print(f"      零方差特征：{len(report['zero_variance'])} 个")
        if report['zero_variance']:
            print(f"      零方差特征列表：{report['zero_variance']}")
        print(f"      异常值总数：{total_outliers}")

        # 打印每个特征的缺失值情况
        print("\n      各特征缺失值统计:")
        for col in features.columns:
            missing_count = features[col].isnull().sum()
            if missing_count > 0:
                print(f"        {col}: {missing_count} 个缺失值 ({missing_count/len(features)*100:.1f}%)")

        return report

    def preprocess(self, features, handle_outliers=True, sigma_threshold=2.0):
        """
        数据预处理

        参数:
            features: DataFrame, 特征数据
            handle_outliers: bool, 是否处理异常值
            sigma_threshold: float, 异常值处理的σ阈值

        返回:
            features_scaled: ndarray, 标准化后的特征
        """
        # 1. 处理缺失值
        print("[3/4] 数据预处理")
        print("      - 处理缺失值（均值填充）...")
        features_imputed = self.imputer.fit_transform(features)
        features_df = pd.DataFrame(features_imputed, columns=features.columns)

        # 2. 处理异常值 (σ截断法)
        if handle_outliers:
            print(f"      - 处理异常值 ({sigma_threshold}σ截断)...")
            for col in features_df.columns:
                mean = features_df[col].mean()
                std = features_df[col].std()
                if std > 1e-10:  # 避免除零
                    lower = mean - sigma_threshold * std
                    upper = mean + sigma_threshold * std
                    features_df[col] = features_df[col].clip(lower, upper)

        # 3. 特征标准化 (Z-score)
        print("      - 特征标准化 (Z-score)...")
        features_scaled = self.scaler.fit_transform(features_df)

        # 打印标准化后的统计信息
        features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)
        print("\n      标准化后统计:")
        print(f"        整体均值：{features_scaled.mean():.2e}")
        print(f"        整体方差：{features_scaled.var():.2e}")

        print("\n      各特征标准化后统计:")
        for i, col in enumerate(features.columns):
            col_mean = features_scaled_df[col].mean()
            col_std = features_scaled_df[col].std()
            print(f"        {col}: 均值={col_mean:.3f}, 标准差={col_std:.3f}")

        return features_scaled

    def fit_transform(self, file_path, handle_outliers=True, sigma_threshold=2.0):
        """
        完整预处理流程

        参数:
            file_path: CSV 文件路径
            handle_outliers: bool, 是否处理异常值
            sigma_threshold: float, 异常值处理的σ阈值

        返回:
            features_scaled: ndarray, 标准化后的特征
            quality_report: dict, 质量检查报告
        """
        # 加载并选择数据
        features = self.load_and_select_data(file_path)

        # 质量检查
        quality_report = self.check_data_quality(features)

        # 预处理
        features_scaled = self.preprocess(features, handle_outliers, sigma_threshold)

        print(f"\n[OK] 数据预处理完成")
        print(f"      输入特征: {len(self.selected_columns)} 个")
        print(f"      输出形状: {features_scaled.shape}")

        return features_scaled, quality_report


def load_and_preprocess_manual(file_path, feature_columns, handle_outliers=True, sigma_threshold=2.0):
    """
    手动特征选择的数据加载和预处理便捷函数

    参数:
        file_path: CSV 文件路径
        feature_columns: list, 用户指定的特征列名列表
        handle_outliers: bool, 是否处理异常值
        sigma_threshold: float, 异常值处理的σ阈值

    返回:
        features: ndarray, 标准化后的特征
        preprocessor: ManualFeaturePreprocessor, 预处理器对象
        report: dict, 质量检查报告
    """
    preprocessor = ManualFeaturePreprocessor(feature_columns)
    features, report = preprocessor.fit_transform(file_path, handle_outliers, sigma_threshold)

    return features, preprocessor, report


if __name__ == "__main__":
    # 测试示例
    file_path = r"C:\Users\Maple\.openclaw\workspace\OGLCM-AKBO\TZ1H_texture_logging.csv"

    # 用户手动指定要使用的特征
    selected_features = ['CON_SUB_DYNA', 'DIS_SUB_DYNA', 'HOM_SUB_DYNA', 'ENG_SUB_DYNA']

    print("="*80)
    print("手动特征选择数据预处理器测试")
    print("="*80)
    print(f"指定特征: {selected_features}")
    print("="*80)

    try:
        features, preprocessor, report = load_and_preprocess_manual(
            file_path=file_path,
            feature_columns=selected_features,
            handle_outliers=True,
            sigma_threshold=2.0
        )

        print("\n" + "="*80)
        print("预处理结果:")
        print(f"  特征数据形状：{features.shape}")
        print(f"  使用的特征：{preprocessor.selected_columns}")
        print("="*80)

    except ValueError as e:
        print(f"错误: {e}")
        print("请检查指定的特征列名是否在数据中存在。")