# %%
##载入相应库函数
from features.feature_engineering import engineer_features
from tools import find_project_root, load_features,  cluster_downsample
from catboost import CatBoostClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import glob

# 
root_path = find_project_root()

# %%
# %% 添加 logging 设置（放在最顶部导入区域附近）
import logging
import os
from datetime import datetime

# 创建 logs 目录
log_dir = os.path.join(str(root_path), 'result/logs')
os.makedirs(log_dir, exist_ok=True)

# 设置日志文件名
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(log_dir, f'train_{timestamp}.log')

# 配置 logging
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

# %% 
## 读取spark的数据
parquet_files = glob.glob(str(root_path) + '/' + "data/raw2/*.parquet")
df = pd.concat([pd.read_parquet(file) for file in parquet_files], ignore_index=True)
print("读取spark数据完成!")

# %%
# 筛选df里面时间小于2025-07-15 00:00:00.000的数据
train_df = df[df.req_time < "2025-07-15 00:00:00.000"]
test_df = df[df.req_time >= "2025-07-15 00:00:00.000"]
print(train_df.shape)
print("数据筛选完成！")

# %% 
##导入相关数据
root_path = find_project_root()
# encode_categorical_method = 'label'  # or 'one-hot' for one-hot encoding
all_features = load_features(str(root_path) + '/' + 'src/features/features.txt')
# df = pd.read_excel(str(root_path) + '/' + 'data/raw/output2.xlsx')
# df2 = pd.read_excel(str(root_path) + '/' + 'data/raw/output3.xlsx')
# df = pd.concat([df,df2], axis=0)
df_fake_train = train_df[all_features.keys()].copy()
df_fake_test = test_df[all_features.keys()].copy()

# %%
# 处理相关数据,增加label
boolean_features = ['is_tablet','nfc_support', 'is_banner','is_video','is_weekday']
for feature in boolean_features:
    df_fake_train[feature] = df_fake_train[feature].astype('boolean')
    df_fake_test[feature] = df_fake_test[feature].astype('boolean')
df_fake_train['is_fake'] = ((df_fake_train['conv_flag'] == 0) & df_fake_train['conv_time'].notna()).astype(int)
df_fake_test['is_fake'] = ((df_fake_test['conv_flag'] == 0) & df_fake_test['conv_time'].notna()).astype(int)
print("构造fake数据完成!")

# %% 
## 特征的提取
df_fake_train_copy = df_fake_train.copy()
## 提取特征
df_fake_train = engineer_features(df_fake_train, df_fake_train_copy, all_features)
df_fake_test = engineer_features(df_fake_test, df_fake_train_copy, all_features)
print("特征提取完成!")

## 存储提取完之后的数据,以便下次直接加载
df_fake_train.to_parquet(str(root_path)+ "/data/processed/df_fake_train.parquet")
df_fake_test.to_parquet(str(root_path)+ "/data/processed/df_fake_test.parquet")

# %%
# 提取处理好的数据
df_fake_train = pd.read_parquet(str(root_path)+ "/data/processed/df_fake_train.parquet")
df_fake_test = pd.read_parquet(str(root_path)+ "/data/processed/df_fake_test.parquet")

# %%
## 样本均衡化处理
# 分别统计正负样本的数量
negative_count = len(df_fake_train[df_fake_train['is_fake'] == 0])
positive_count = len(df_fake_train[df_fake_train['is_fake'] == 1])
# 对负样本进行随机采样，采样比例为1:10
negative_ratio = 1
positive_ratio = 1
df_fake_train_negative_sample = df_fake_train[df_fake_train['is_fake'] == 0].sample(frac=negative_ratio, replace= False, random_state=42)
# 存储另外90%的负样本
df_fake_train_negative_resiual = df_fake_train[df_fake_train['is_fake'] == 0].drop(df_fake_train_negative_sample.index)
# 对正样本进行随机的过采样，采样到原样本的两倍
df_fake_train_positive_sample = df_fake_train[df_fake_train['is_fake'] == 1].sample(frac = positive_ratio,replace=False, random_state=42)
# 计算采样后的正负样本比例
weight_ratio = df_fake_train_negative_sample.shape[0] / df_fake_train_positive_sample.shape[0]
# 拼接样本
df_fake_train_end = pd.concat([df_fake_train_negative_sample, df_fake_train_positive_sample], ignore_index=True)

logging.info("正样本采样比例：%d, 负样本采样比例：%d", positive_ratio, negative_ratio)


 # %% 
## 划分训练集和测试集
# 取除了is_fake列的
y_train = df_fake_train_end['is_fake']
df_fake_train_end = df_fake_train_end.drop(columns=['is_fake'])
y_test = df_fake_test['is_fake']
df_fake_test_end = df_fake_test.drop(columns=['is_fake'])
# Y = df_fake['is_fake']
# df_fake_train_end, df_fake_test_end, y_train, y_test = train_test_split(X,Y,  test_size=0.2, random_state=42)
print("划分训练集和测试集完成!")


# %%
# df_fake的所有特征存储为txt

with open('output.txt', 'w') as f:
    for item in df_fake_train_end.columns:
        f.write(f"{item}\n")

# %%
# 制定用于训练的特征
# drop_features = ['click_time', 'click_flag', 'conv_flag', 'conv_time', 'device_model', 'device_make', 'imp_size', 'size']
drop_features = ['click_time', 'click_flag', 'conv_flag', 'conv_time', 'device_model', 'device_make', 'imp_size', 'size']


# 用于训练的特征列
all_proposed_features = [x for x in df_fake_train_end.columns if x not in drop_features]



# 制定类别特征
cate_features = pd.read_csv(str(root_path)+ '/' + 'src/features/cate_features2.txt', header=None, names=['feature_name'])
cate_features = cate_features['feature_name'].tolist()


X_train = df_fake_train_end[all_proposed_features]

# %%
## 模型训练前，进行数据的聚类降采样处理
from tools import cluster_downsample
X_train, y_train = cluster_downsample(X_train, y_train, cate_features=cate_features, target_ratio = 20,  random_state=42)

# %% 
## 统计X_train中正样本和负样本的分布
neg_count = y_train.value_counts()[0]
pos_count = y_train.value_counts()[1]
weight_ratio = neg_count / pos_count

logging.info("当前模型训练数据集大小：%d, 正样本数：%d， 负样本数：%d", len(X_train), positive_count, negative_count)
print("正样本数量：", pos_count)
print("负样本数量：", neg_count)
print("正样本权重：", weight_ratio)

print("模型训练开始：")
# %%
# 制定模型
model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.05,
    depth=6,
    class_weights=[1, weight_ratio],
    loss_function='Logloss',
    eval_metric='AUC',
    cat_features= cate_features,
    verbose=True,
    task_type='CPU',
    random_seed=42
)
logging.info("模型参数配置：%s", model.get_params())
model.fit(X_train,y_train, early_stopping_rounds=50)
print("第一阶段模型训练完成!")

 # %% 测试在训。集上的效果
# 测试模型在另外90%训练集上的效果
y_train_negative_resiual= df_fake_train_negative_resiual['is_fake']
X_train_negative_resiual= df_fake_train_negative_resiual[all_proposed_features]
y_pred_train_negative_resiual = model.predict(X_train_negative_resiual)
y_pre_train_label= (y_pred_train_negative_resiual>=0.5).astype(int)
# %%
logger.info("训练集剩余样本上的预测结果: 正确预测数：%d / %d", 
            (y_pre_train_label == 1).sum(), len(y_train_negative_resiual))
# 测试模型在原有训练集上的效果
from sklearn.metrics import confusion_matrix, accuracy_score
pred_train_labels = model.predict(X_train)
acc_train = accuracy_score(y_train, pred_train_labels)

logger.info("训练集混淆矩阵:\n%s", confusion_matrix(y_train, pred_train_labels))
logger.info("训练集准确率: %.4f", acc_train)

# %%
model.save_model(str(root_path) + '/'+ 'result/model/model.json', format = 'json')
print("模型保存成功!")

# %%
# 输出变量重要性结果
print(sorted(model.get_feature_importance(), reverse=True))


# %%
# 进行预测
# 分类标签
X_test = df_fake_test_end[all_proposed_features]
pred_labels = model.predict(X_test)

# 输出混淆矩阵
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, pred_labels))

# 输出准确率
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, pred_labels))

# 输出AUC
from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, pred_labels))
print(model.get_feature_importance())
print(model.get_feature_importance(prettified=True))

logger.info("开始在测试集上进行预测")
logger.info("测试集混淆矩阵:\n%s", confusion_matrix(y_test, pred_labels))
logger.info("测试集准确率: %.4f", accuracy_score(y_test, pred_labels))
logger.info("测试集 AUC: %.4f", roc_auc_score(y_test, pred_labels))


# 作出ROC曲线
# %%
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
fpr, tpr, thresholds = roc_curve(y_test, pred_labels)
plt.plot(fpr, tpr)
plt.show()



# %%
