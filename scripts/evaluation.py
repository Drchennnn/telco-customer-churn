import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 加载特征工程后的数据
data_path = '../data/engineered_telco_data.csv'
df = pd.read_csv(data_path)

# 分割数据为特征和目标变量
X = df.drop('Churn', axis=1)
y = df['Churn']

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练最佳模型（随机森林）
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 评估
print('分类报告:')
print(classification_report(y_test, y_pred))

print('混淆矩阵:')
print(confusion_matrix(y_test, y_pred))
