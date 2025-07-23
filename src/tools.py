import os 
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from kmodes.kprototypes import KPrototypes

# 定义一个函数来查找项目根目录
def find_project_root(start_path=None):
    if start_path is None:
        start_path = os.getcwd()  # 获取当前工作目录
    start_path = Path(start_path).resolve()  # 确保是绝对
    current_path = start_path
    while current_path != current_path.parent:  # 到了根目录还没找到就停
        # 判断是否有标志性文件夹，比如 data 和 src
        if (current_path / 'data').exists() and (current_path / 'src').exists():
            return current_path
        current_path = current_path.parent

    raise FileNotFoundError("没找到项目根目录")

# Extract features from a features.txt, which contains feature names and its types
def load_features(features_file_path):
    features = pd.read_csv(features_file_path, header=None, names=['feature_name'])
    all_features = {}
    for feature in features['feature_name']:
        feature_split = feature.split(' ')
        if len(feature_split) > 1:
            feature_name = feature_split[0]
            feature_type = feature_split[1]
            all_features[feature_name] = feature_type
        else:
            all_features[feature_split[0]] = 'unknown'
    return all_features

# dowmsample from data
def cluster_downsample(X, y, cate_features=None, target_ratio = 1.2,  random_state=42):

    
    """
    使用 K-Prototypes 对负样本进行聚类降采样，使得正负样本接近平衡。
    
    参数:
        X (pd.DataFrame): 特征数据（包含数值型 + 类别型特征）
        y (pd.Series): 标签，0=负样本，1=正样本
        cate_features (list[str]): 类别特征列名列表
        random_state (int): 随机种子
        cluster_ratio (float): 聚类簇数设置为正样本数量 * cluster_ratio

    返回:
        X_balanced (pd.DataFrame): 平衡后的特征集
        y_balanced (pd.Series): 平衡后的标签集
    """

    # 验证输入
    if cate_features is None:
        cate_features = []

    # 只提取必要的负样本索引（不复制 DataFrame）
    pos_idx = y[y == 1].index
    neg_idx = y[y == 0].index

    n_pos = len(pos_idx)
    n_clusters = min(len(neg_idx), int(n_pos * target_ratio))

    # 获取类别与数值型特征列
    numeric_features = [col for col in X.columns if col not in cate_features]
    used_columns = cate_features + numeric_features

    # 构造用于聚类的负样本数据（不影响原始 X）
    X_neg_for_cluster = X.loc[neg_idx, used_columns].copy()

    if cate_features:
        X_neg_for_cluster[cate_features] = X_neg_for_cluster[cate_features].fillna("missing")

    # 转为 numpy（必须 object 类型）
    X_neg_np = X_neg_for_cluster.to_numpy(dtype=object)


    # # 从 X 中直接构造用于聚类的 numpy 数据（不复制多份）
    # X_neg_np = X.loc[neg_idx, used_columns].to_numpy(dtype=object)

    # 获取类别列在矩阵中的位置索引
    categorical_cols_idx = [used_columns.index(col) for col in cate_features]

    # 聚类
    kproto = KPrototypes(n_clusters=n_clusters, init='Cao', verbose=0, random_state=random_state)
    cluster_labels = kproto.fit_predict(X_neg_np, categorical=categorical_cols_idx)

    # 构建 cluster → 样本索引 映射（用数组更快）
    cluster_to_indices = {i: [] for i in range(n_clusters)}
    for idx, cluster_id in zip(neg_idx, cluster_labels):
        cluster_to_indices[cluster_id].append(idx)

    # 每个簇等量采样
    samples_per_cluster = max(1, n_pos // n_clusters)
    sampled_neg_indices = []
    rng = np.random.default_rng(seed=random_state)

    for cluster_id, indices in cluster_to_indices.items():
        if len(indices) > 0:
            sampled = rng.choice(indices, size=min(len(indices), samples_per_cluster), replace=False)
            sampled_neg_indices.extend(sampled)

    # 构造新数据集（只合并最终索引）
    selected_idx = list(pos_idx) + sampled_neg_indices
    X_balanced = X.loc[selected_idx]
    y_balanced = y.loc[selected_idx]

    # 打乱
    X_balanced = X_balanced.sample(frac=1.0, random_state=random_state)
    y_balanced = y_balanced.loc[X_balanced.index]

    print(f"[INFO] 正样本: {n_pos} | 采样后负样本: {len(sampled_neg_indices)} | 聚类簇数: {n_clusters}")
    return X_balanced, y_balanced

