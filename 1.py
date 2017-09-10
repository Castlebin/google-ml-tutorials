from sklearn import tree

# 特征
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
# 标签
labels = [0, 0, 1, 1]
# 使用决策树进行学习
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

# 使用经过训练数据训练后的分类器对新的特征值进行预测
print(clf.predict([[145, 0]]))
