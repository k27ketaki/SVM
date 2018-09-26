from sklearn import datasets
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

def plot(X,y,clf,title):
	#Creating mesh to plot in
	x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
	y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),np.arange(y_min, y_max, 0.2))

	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	fig, ax = plt.subplots()
	ax.contourf(xx, yy, Z, cmap=plt.cm.plasma, alpha=0.8)
	for i in range(len(y)):
		ax.scatter(X[i,0], X[i,1], c=color[y[i]], marker=marker[y[i]],s=30,edgecolors='k')
	ax.set_xlim(xx.min(), xx.max())
	ax.set_ylim(yy.min(), yy.max())
	ax.set_xlabel('Petal length')
	ax.set_ylabel('Petal width')
	ax.set_title(title)
	plt.show()
	print("Accuracy of SVC with ",title,"is::",clf.score(X,y)*100,"%")

# import iris data to model Svm classifier
iris_dataset = datasets.load_iris()

print("Iris data set Description :: ", iris_dataset['DESCR'])

#We are using petal length and petal width
X = iris_dataset.data[:, 2:4]
y = iris_dataset.target
color = ['r','g','b']
marker = ['o','^','*']
for i in range(len(y)):
	plt.scatter(X[i,0], X[i,1], c=color[y[i]], marker=marker[y[i]])
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.title('Petal Width & Length')
plt.show()

#Regularization parameter
C = 1.0
 
#SVC with linear kernel
svc = svm.SVC(kernel='linear', C=C).fit(X, y)
#SVC with RBF kernel
rbf_svc = svm.SVC(kernel='rbf', gamma=0.5, C=C).fit(X, y)
#SVC with polynomial (degree 3) kernel
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)

plot(X,y,svc,"Linear kernel")
plot(X,y,rbf_svc,"RBF kernel")
plot(X,y,poly_svc,"Polynomial(degree 3) kernel")