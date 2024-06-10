import pandas as pd

# 加载预处理数据
data_path = '../data/preprocessed_telco_data.csv'
df = pd.read_csv(data_path)

# 创建 tenure 分组
df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 60, 72], labels=['0-12', '12-24', '24-48', '48-60', '60-72'])

# 保存特征工程后的数据
df.to_csv('../data/engineered_telco_data.csv', index=False)
