# OGLCM-AKBO

**定向灰度共生矩阵 (OGLCM) + 贝叶斯优化自适配 K-means (AKBO)**

基于测井图像纹理特征的页岩微相自动表征方法

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version 2.1](https://img.shields.io/badge/version-2.1-yellow.svg)](https://github.com/bluemap19/OGLCM-AKBO)

---

## 📖 项目简介

OGLCM-AKBO 是一种**无监督聚类算法**，用于基于测井图像纹理特征自动识别页岩微相。

**核心创新：**

1. 使用 OGLCM 提取 28 个纹理特征
2. 基于随机森林特征选择 4 个最优特征
3. 使用贝叶斯优化自动确定最优 K 值
4. 提出 UIndex 复合指标评估聚类质量

**代码实现的功能：**

1. PCA数据降维
2. 构建KMeans的参数空间
3. 使用贝叶斯优化自动确定最优 KMeans 参数值

---

## 🚀 快速开始

### 环境要求

- Python 3.8+
- NumPy >= 1.20.0
- Pandas >= 1.3.0
- Scikit-learn >= 0.24.0
- Matplotlib >= 3.4.0

### 安装依赖

```bash
# 克隆项目
git clone https://github.com/bluemap19/OGLCM-AKBO.git
cd OGLCM-AKBO

# 安装依赖
pip install -r requirements.txt
```

### 运行示例

```bash
# 运行主程序
python main.py
```

**输入：** `TZ1H_texture_logging.csv` (测井纹理特征数据)

**输出：**
- `results/clustering_results.csv` - 聚类结果
- `results/test_report.md` - 详细测试报告（包含完整迭代历史）
- `results/figures/*.png` - 可视化图表

---

## 📊 数据说明

### 输入数据格式

| 列名 | 说明 | 示例 |
|------|------|------|
| DEPTH | 深度 (m) | 2192.21 |
| CON_SUB_DYNA | 对比度_子区域_动态 | 0.523 |
| DIS_SUB_DYNA | 差异性_子区域_动态 | 0.412 |
| HOM_SUB_DYNA | 同质性_子区域_动态 | 0.678 |
| ENG_SUB_DYNA | 熵_子区域_动态 | 0.345 |
| ... | 其他特征 (共 28 个) | ... |

### 最优特征（4 个）

| 特征名 | 地质意义 |
|--------|----------|
| `CON_SUB_DYNA` | 对比度_子区域_动态 - 反映局部纹理变化强度 |
| `DIS_SUB_DYNA` | 差异性_子区域_动态 - 反映子区域灰度差异 |
| `HOM_SUB_DYNA` | 同质性_子区域_动态 - 反映子区域纹理均匀度 |
| `ENG_SUB_DYNA` | 熵_子区域_动态 - 反映纹理复杂度 |

---

## 🔬 核心算法

### UIndex 复合指标

**Doctor 提出的创新公式：**

```
UIndex(K) = 1 / (0.1/SI(K) + DBI(K)/1.0 + 0.01/DVI(K))
```

| 指标 | 说明 | 特性 |
|------|------|------|
| **SI** | 轮廓系数 | 越大越好 |
| **DBI** | Davies-Bouldin Index | 越小越好 |
| **DVI** | Dunn Index | 越大越好 |

**质量评价标准：**
- UIndex > 0.5: 优秀
- UIndex > 0.2: 良好
- UIndex < 0.2: 需改进

### 贝叶斯优化流程

```
初始化 (5 个点) → 高斯过程建模 → EI 采集函数 → 迭代 (30 次) → 最优 K 值
```

---

## 📁 项目结构

```
OGLCM-AKBO/
├── main.py                      # 主程序
├── src/
│   ├── data_loader.py           # 数据加载与预处理
│   ├── akbo_clustering.py       # AKBO 核心算法
│   └── visualization.py         # 可视化模块
├── tests/
│   └── test_akbo.py            # 单元测试
├── docs/
│   ├── 特征选择说明.md           # 特征选择方法
│   ├── 技术路线与算法原理.md     # 算法原理
│   └── 版本 2.1 变更总结.md       # 版本变更
├── results/                     # 输出结果（git 忽略）
├── .gitignore                   # Git 忽略文件
├── requirements.txt             # 依赖包
├── LICENSE                      # 许可证
└── README.md                    # 项目说明
```

---

## 📈 测试结果

### 示例数据（TZ1H 井）

| 指标 | 值 | 评价 |
|------|-----|------|
| **样本数** | 8265 | - |
| **特征数** | 4 | - |
| **最优 K** | 5 | - |
| **UIndex** | 0.9828 | ✅ 优秀 |
| **SI** | 0.4552 | ⚠️ 一般 |
| **DBI** | 0.7643 | ✅ 良好 |
| **DVI** | 0.2987 | ⚠️ 较低 |

### 聚类分布

| Cluster | 样本数 | 占比 |
|---------|--------|------|
| 0 | 2629 | 31.8% |
| 1 | 951 | 11.5% |
| 2 | 1377 | 16.7% |
| 3 | 2543 | 30.8% |
| 4 | 765 | 9.3% |

---

## 📚 文档

- [特征选择说明](docs/特征选择说明.md) - 特征选择方法和地质意义
- [技术路线与算法原理](docs/技术路线与算法原理.md) - 完整算法原理和公式
- [版本 2.1 变更总结](docs/版本 2.1 变更总结.md) - 代码变更说明

---

## 🔧 版本历史

### v2.1 (2026-03-27)
- ✅ 更新 UIndex 计算公式（Doctor 提出的新公式）
- ✅ 移除聚类器中的特征选择功能（已在预处理阶段完成）
- ✅ 使用预设的 4 个最优特征
- ✅ 添加详细迭代历史记录到测试报告

### v2.0 (2026-03-15)
- ✅ 修复 EI 采集函数错误
- ✅ 使用 GMM 计算真实后验概率
- ✅ 添加随机森林特征选择
- ✅ 添加单元测试 (15/15 通过)

---

## 👥 作者

- **Fuhao Zhang (Doctor)**
  - GitHub: [@bluemap19](https://github.com/bluemap19)
  - Email: puremaple19@outlook.com
  - Research Area: Well Logging Engineering, Geological Engineering, Deep Learning

- **Cuka** (OpenClaw AI Assistant)

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

感谢所有为该项目做出贡献的研究人员和技术支持者！

---

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- GitHub Issues: https://github.com/bluemap19/OGLCM-AKBO/issues
- Email: puremaple19@outlook.com

---

*Last Updated: 2026-03-27*
