import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv(r"banknote.txt",sep=",",
                 names = ['var','skewness','kurtosis','entropy','class'])

print(df.head())
fig = plt.figure()
plt.scatter(df['var'],df['skewness'],c=df['class'],alpha=0.5)
plt.xlabel('Variance')
plt.ylabel('Skewness')
plt.show()
X = df.drop(['class'],axis=1)
Y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state=100)
clf_gini = DecisionTreeClassifier(criterion = "gini", max_depth = 3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)

tree.export_graphviz(clf_gini, out_file =r'tree.dot')

clf_entropy = DecisionTreeClassifier(criterion = "entropy", max_depth = 3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)

tree.export_graphviz(clf_entropy, out_file =r'tree1.dot')
y_pred_gini = clf_gini.predict(X_test)
y_pred_entropy = clf_entropy.predict(X_test)
print("Accuracy is ",accuracy_score(y_test, y_pred_gini)*100)
print("Accuracy is ",accuracy_score(y_test, y_pred_entropy)*100)
