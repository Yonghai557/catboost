from tools import find_project_root, load_features
from features.feature_engineering import engineer_features
import pandas as pd
from catboost import CatBoostClassifier


# 测试数据的路径
root_path = find_project_root()

# 载入测试数据
df = pd.read_csv(str(root_path) + '/data/test_data.csv')

# 载入所有特征及其格式
all_features = load_features(str(root_path) + '/' + 'src/features/features.txt')

df = df[all_features.keys()].copy()

# %%
## 特征处理数据的处理
# 首先处理bool型变量
boolean_features = ['is_tablet','nfc_support', 'is_banner','is_video','is_weekday']
for feature in boolean_features:
    df[feature] = df[feature].astype('boolean')

# 提取参考数据的
base_data = pd.read_csv(str(root_path) + '/data/base_data.csv')

# 进行特征提取
df = engineer_features(df, base_data, all_features)

#筛选得到最后的输入
drop_features = ['click_time', 'click_flag', 'conv_flag', 'conv_time', 'device_model', 'device_make', 'imp_size', 'size']
all_proposed_features = [x for x in df.columns if x not in drop_features]

df = df[all_proposed_features]

# 载入模型
model = CatBoostClassifier()
model.load_model(str(root_path) + '/' + 'result/model/model.cbm')

# 预测得到结果
preds = model.predict_proba(df)[:,1]


