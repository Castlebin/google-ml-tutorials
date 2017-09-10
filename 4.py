from sklearn import datasets

iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
# 将数据集划分为两部分，一部分用于训练，一部分用于测试(0.5表示均分)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

print("X_train: ", X_train)
print("y_train: ", y_train)
print("X_test: ", X_test)
print("y_test: ", y_test)

# 选择一个决策树作为分类器
from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()

# 使用另一种分类器
# 使用KNeighborsClassifier
# from sklearn.neighbors import KNeighborsClassifier
# my_classifier = KNeighborsClassifier()

# 使用测试数据进行训练
my_classifier.fit(X_train, y_train)

# 使用训练好的模型进行预测
predictions = my_classifier.predict(X_test)
print("predictions: ", predictions)

# 将预测结果与测试数据进行比较，观察预测的准确性
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))

