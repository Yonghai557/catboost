# %%
##  load necessary libraries
import pandas as pd
import numpy as np
import os
from pathlib import Path
import ipaddress


# %%
# define functions for feature engineering
# define a function to encode categorical variables
def encode_categorical_variable(df1,df2,feature_name, method='label'):
    if method == 'label':
        cate1 = df1[feature_name].astype('category').cat
        cate2 = df2[feature_name].astype('category').cat

        df[feature_name] = df[feature_name].astype('category').cat.codes # Label encoding
    elif method == 'one-hot':
        df = pd.get_dummies(df, columns=[feature_name], prefix=feature_name, drop_first=True) # drop_first = True remove the first category to avoid dummy variable trap
    else:
        raise ValueError("Unknown encoding method: {}".format(method))
    return df
def encode_categorical_with_rare(df1, df2, feature_name, rare_threshold=10):
    """
    使用 df_train 的统计信息对 df_test[feature_name] 编码，
    将 df_test 中不在 df_train 中的类别或低频类别替换为 '__rare__'，再做 Label Encoding。
    
    参数:
        df_train (pd.DataFrame): 训练集
        df_test (pd.DataFrame): 测试集
        feature_name (str): 要编码的特征名
        rare_threshold (int): 稀有类别阈值，默认为 10
    
    返回:
        pd.DataFrame: 处理后的测试集
    """
    # 统计训练集中每个类别的频次
    feature_counts = df1[feature_name].value_counts(dropna=False)
    # 定义替换函数
    def replace_with_rare(x):
        if x in feature_counts and feature_counts[x] >= rare_threshold:
            return x
        else:
            return '__rare__'

    # 应用替换逻辑
    df2.loc[:, feature_name] = df2[feature_name].apply(replace_with_rare)

    # 获取所有可能的类别（包括 '__rare__'）
    unique_categories = list(feature_counts[feature_counts >= rare_threshold].index) + ['__rare__']

    # 创建类别到编码的映射字典
    category_to_code = {cat: idx for idx, cat in enumerate(unique_categories)}

    # Label Encoding
    df2.loc[:, feature_name] = df2[feature_name].map(category_to_code)

    return df2


# bucketize release_msrp feature
def bucketize_release_msrp(df, feature_name, bins=None):
    if bins is None:
        bins = [0,1000, 3000, 5000,8000,np.inf]  # 默认分桶
    labels = [1, 2, 3, 4, 5]
    df[feature_name] = pd.cut(df[feature_name], bins=bins, labels=labels, include_lowest=True)
    return df
# bucketize imp_bid_floor feature
def bucketize_imp_bid_floor(df, feature_name, bins=None):
    if bins is None:
        bins = [0, 0.5, 1, 2, 5, 10, 20, float('inf')]  # 默认分桶
    labels = [1, 2, 3, 4, 5,6,7]
    df[feature_name] = pd.cut(df[feature_name], bins=bins, labels=labels, include_lowest=True)
    return df
# bucketize lattitude or longitude feature
def bucketize_latitude_longitude(df, feature_name, bins=None):
    if feature_name == 'latitude':
        if bins is None:
            bins = [-90,-70,-50,-30, -10, 0, 10, 30, 50, 70, 90]
    else:
        if bins is None:
            bins = [-180,-140,-100,-60, -20, 0, 20, 60, 100, 140, 180]
    labels = list(range(1, len(bins)))
    df[feature_name] = pd.cut(df[feature_name], bins=bins, labels=labels, include_lowest=True)
    return df
# bucketize count features
def bucketize_count_features(df, feature_name, bins=None):
    if bins is None:
        bins = [0,1,2,4,8,16,int(1e6)]  # 默认分桶
    labels = list(range(1, len(bins)))
    df[feature_name] = pd.cut(df[feature_name], bins=bins, labels=labels, include_lowest=True)
    return df

# define a function to address numerical variables
def process_numerical_variable(df, feature_name):

    count_cols = [
    "open_1d", "open_3d", "open_7d", "open_all",
    "view_1d", "view_3d", "view_7d", "view_all",
    "atw_1d", "atw_3d", "atw_7d", "atw_all",
    "atc_1d", "atc_3d", "atc_7d", "atc_all",
    "pur_3d", "pur_7d", "pur_15d", "pur_all",
    "imps_1d", "imps_3d", "imps_7d", "imps_30d",
    "clicks_1d", "clicks_3d", "clicks_7d", "clicks_30d"
]
    
    # 分步处理不同的数值型变量
    if feature_name == 'release_msrp':
        # 可以对这个变量进行分桶处理
        df = bucketize_release_msrp(df, feature_name)
    elif feature_name == 'screen_width' or feature_name == 'screen_height':
        # 对屏幕的某些尺寸进行处理(需处理)
        df = df
    elif feature_name == 'imp_width' or feature_name == 'imp_height':
        # 对广告的某些尺寸进行处理(需处理)
        df = df
    elif feature_name == 'imp_bid_floor':
        # 对广告的最低出价进行处理
        df = bucketize_imp_bid_floor(df, feature_name)
    elif feature_name == 'latitude' or feature_name == 'longitude':
        # 对经纬度进行分桶处理
        df = bucketize_latitude_longitude(df, feature_name)
    # elif feature_name in count_cols:
    #     # 对计数型变量进行分桶处理
    #     df = bucketize_count_features(df, feature_name)
    return df


# %%
## load features from features.txt


# 对IP字段进行处理
def extract_ip_features(ip):
    if pd.isna(ip):
        return {
        'ip_type': np.nan,
        'is_private': np.nan,
        'ip_prefix': np.nan
        }
    result = {
        'ip_type': 'unknown',
        'is_private': None,
        'ip_prefix': None
    }
    try:
        ip_address = ipaddress.ip_address(ip)
        result['ip_type'] = ip_address.version
        result['is_private'] = ip_address.is_private
        if result['ip_type']==4:
            # 取前缀并用'.'连接
            result['ip_prefix'] = '.'.join(ip.split('.')[:3])
        elif result['ip_type']==6:
            result['ip_prefix'] = ':'.join(ip.split(':')[:3])
    except Exception as e:
        print(f"Error processing IP address '{ip}': {e}")
    return result

# %%
#  从df中筛选出需要的特征数据
def engineer_features(df2, df1, all_features, encode_categorical=True):
    '''
    use baseline data df1 to engineer features for df2.
    
    Parameters
    df2 : DataFrame, data needs to be processed
    df1 : DataFrame, baseline data for feature engineering
    all_features : dict
    encode_categorical : bool, optional
    
    Returns
    df2: DataFrame'''
    ## constrcut new features
    # 计算广告的宽高比
    # df_fake['imp_aspect_ratio'] = df_fake['imp_width'] / df_fake['imp_height']
    df2['imp_aspect_ratio'] = df2['imp_width'] / df2['imp_height']
    # 计算屏幕的宽高比
    # df_fake['screen_aspect_ratio'] = df_fake['screen_width'] / df_fake['screen_height']
    df2['screen_aspect_ratio'] = df2['screen_width'] / df2['screen_height']
    # 计算广告的宽高比与屏幕的宽高比的差
    # df_fake['aspect_ratio_diff'] = df_fake['imp_aspect_ratio'] - df_fake['screen_aspect_ratio']
    df2['aspect_ratio_diff'] = df2['imp_aspect_ratio'] - df2['screen_aspect_ratio']


    # %%
    ## feature engineering
    # 对device_hwv变量，normalized_device_model, carrier变量，处理见下
    all_features_copy = all_features.copy()
    for feature_name, feature_type in all_features_copy.items():
        if feature_name == 'device_ip':
            df1_features = df1[feature_name].apply(extract_ip_features).apply(pd.Series)
            df2_features = df2[feature_name].apply(extract_ip_features).apply(pd.Series)
            df1 = pd.concat([df1, df1_features], axis=1)
            df2 = pd.concat([df2, df2_features], axis=1)
            all_features['ip_type'] = 'category'
            all_features['ip_prefix'] = 'category'
            df2.drop(columns=[feature_name], inplace=True)
            
        elif feature_name == 'device_hwv' or feature_name == 'normalized_device_model' or feature_name == 'carrier' or feature_name == 'isp' or feature_name == 'publisher_app_id' or feature_name == 'creative_id' or feature_name == 'advertiser_app_id' or feature_name == 'region' or feature_name == 'city' or feature_name == 'areacode':
            feature_counts = df1[feature_name].value_counts(dropna=False)
            # 将df2中出现但是df1中没有的feature_name频数设置为0
            df2[feature_name + '_freq'] = df2[feature_name].apply(lambda x: int(np.log1p(feature_counts.get(x, 0))))
            # 构造特征替换，小于rare_threshold的频数的类别设置__rare__
            rare_threshold = 10
            df2[feature_name] = df2[feature_name].apply(
                lambda x: x if feature_counts.get(x, 0) >= rare_threshold else '__rare__'
                )
        elif feature_name == 'req_time':
            # req_time的格式是2025-07-16 01:47:55.291, 从格式中分别提取dow，hr, is_weekday等特征
            dt_series = pd.to_datetime(df2[feature_name], errors='coerce')
            df2[feature_name + '_dow'] = dt_series.dt.dayofweek
            df2[feature_name + '_hr'] = dt_series.dt.hour
            df2[feature_name + '_is_weekday'] = df2[feature_name + '_dow'].apply(lambda x: x < 5 if not pd.isna(x) else np.nan)
            df2.drop(columns=[feature_name], inplace=True)
        elif feature_name == 'release_date':
            month_map = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
    }
            col = df2[feature_name]
             # 提取年份和月份
            df2[feature_name + '_year'] = (
                col.str.extract(r'(\d{4})')[0]
                .astype(str)  # pandas 的 Nullable Integer 类型
            )

            df2[feature_name + '_month'] = (
                col.str.extract(r'_([a-zA-Z]+)')[0]
                .str.lower()
                .map(month_map)
                .astype('Int64').astype(str)  # 防止 map 失败变成 NaN 后出错
    )
            df2.drop(columns=[feature_name], inplace=True)  # 删除原始的时间


    # 重新遍历历 all_features字典，对不同类型的变量做编码
    for feature_name, feature_type in all_features.items():
        if feature_name in df2.columns:
            if feature_type == 'numerical':
                # 对数值型变量进行处理
                # df2 = process_numerical_variable(df2, feature_name)
                # 缺失值填充为-1
                df2[feature_name].fillna(-1, inplace=True)
            elif feature_type == 'category':
                df2[feature_name] = df2[feature_name].astype(str)
                
            #     df2 = encode_categorical_with_rare(df1, df2, feature_name)
            elif feature_type == 'binary':
                # 将二值变量处理成int
                df2[feature_name] = df2[feature_name].astype(str)

    return df2
    return df