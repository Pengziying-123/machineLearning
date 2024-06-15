import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score

# 读取训练集和测试集数据
train_data = pd.read_csv("train_set.csv", sep='\t')
test_data = pd.read_csv("test_a.csv", sep='\t')

# 输出训练集的形状
print(train_data.shape)

# 输出训练集的前5行
print(train_data[:5])

# 使用TfidfVectorizer对文本数据进行特征提取，得到TF-IDF矩阵
tf_idf = TfidfVectorizer(max_features=2000).fit(train_data['text'].values)
train_tfidf = tf_idf.transform(train_data['text'].values)

# 输出训练集的TF-IDF矩阵形状
print(train_tfidf.shape)

# 对测试集数据进行特征提取
test_tfidf = tf_idf.transform(test_data['text'].values)

# 使用岭回归进行建模
clf = RidgeClassifier()
clf.fit(train_tfidf, train_data['label'].values)

# 输出模型信息
print(RidgeClassifier())

# 计算模型在验证集上的F1值
val_pred = clf.predict(train_tfidf[10000:])
print(f1_score(train_data['label'].values[10000:], val_pred, average='macro'))

# 对测试集数据进行预测，并将结果保存到CSV文件中
df = pd.DataFrame()
df['label'] = clf.predict(test_tfidf)
df.to_csv('submit.csv', index=None)
