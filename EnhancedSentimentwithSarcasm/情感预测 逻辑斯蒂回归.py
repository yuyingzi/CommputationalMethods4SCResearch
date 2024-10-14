import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("IMDB Dataset.csv")

import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
def preprocess_text(text):
    # 清除HTML标签和特殊字符
    clean_text = re.sub('<.*?>', '', text)
    clean_text = re.sub('[^a-zA-Z]', ' ', clean_text)
    
    # 将文本转换为小写
    clean_text = clean_text.lower()
    
    # 切割成单词
    words = clean_text.split()
    
    # 去除停用词
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # 对文本进行词干化处理
    stemmer = SnowballStemmer('english')
    words = [stemmer.stem(word) for word in words]
    
    # 将单词重新组合成文本
    clean_text = ' '.join(words)
    
    return clean_text

data['review'] = data['review'].apply(preprocess_text)

# 划分训练集和测试集
train_texts, test_texts, train_labels, test_labels = train_test_split(data['review'], data['sentiment'], test_size=0.2, random_state=42)
# 创建TF-IDF向量化器
vectorizer = TfidfVectorizer()
# 在训练集上进行特征提取
train_features = vectorizer.fit_transform(train_texts)
# 在测试集上进行特征提取
test_features = vectorizer.transform(test_texts)

# 创建朴素贝叶斯分类器
model = LogisticRegression()
# 在训练集上训练模型
model.fit(train_features, train_labels)

# 在测试集上进行预测
pred_labels = model.predict(test_features)
# 计算准确率和其他评估指标
accuracy = accuracy_score(test_labels, pred_labels)
report = classification_report(test_labels, pred_labels)
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")