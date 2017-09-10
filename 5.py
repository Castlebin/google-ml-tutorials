from scipy.spatial import distance

# 计算欧式距离的方法
def euc(a, b):
    return distance.euclidean(a, b)

# 编写一个我们自己的分类器
class ScrappyKNN():
    def fit(self, X_train, y_train):
        # 保存训练数据在模型中
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []

        # 随机挑选一个标签作为预测结果
        for row in X_test:
            # 使用最近邻的标签作为自己的标签输出
            label = self.closest(row)
            predictions.append(label)

        return predictions

    # 找出最近邻，将它的标签返回
    def closest(self, row):
        # 将与第0个点的距离作为初始值
        best_dist = euc(row, self.X_train[0])
        # 将第0个点的标签作为初始值
        best_index = 0

        # 迭代所有的点，找出最近邻的点
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        # 将最近邻点的标签作为输出
        return self.y_train[best_index]


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

my_classifier = ScrappyKNN()

# 使用测试数据进行训练
my_classifier.fit(X_train, y_train)

# 使用训练好的模型进行预测
predictions = my_classifier.predict(X_test)
print("predictions: ", predictions)

# 将预测结果与测试数据进行比较，观察预测的准确性
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))

