import pandas as pd


# A Summary of the PCA Approach

'''
1. Standardize the data.
2. Obtain the Eigenvectors and Eigenvalues from the covariance matrix or correlation matrix.
3. Sort eigenvalues in descending order and choose the k-eigenvectors that correspond to the  
   k largest eigenvalues where k is the number of dimensions of the new feature subspace (k≤d)
4. Construct the projection matrix W from the selected k eigenvectors.
5. Transform the original dataset X via W to obtain a k-dimanesional feature space Y

'''


feature_dict = {i:label for i,label in zip(
                range(4),
                  ('sepal length in cm',
                  'sepal width in cm',
                  'petal length in cm',
                  'petal width in cm', ))}


irisDF = pd.read_csv('iriss.csv')


irisDF.columns = [l for i,l in sorted(feature_dict.items())] + ['class label']

irisDF.dropna(how="all", inplace=True) # to drop the empty line at file-end

print (irisDF.tail()   )
irisDF.to_excel('iris.xlsx') 
    
    
# Since it is more convenient to work with numerical values, we will use the LabelEncode 
# from the scikit-learn library to convert the class labels into numbers: 1, 2, and 3    

from sklearn.preprocessing import LabelEncoder

X = irisDF.iloc[:,0:4].values
y = irisDF['class label'].values



enc = LabelEncoder()
label_encoder = enc.fit(y)
y = label_encoder.transform(y) + 1

label_dict = {1: 'Setosa', 2: 'Versicolor', 3:'Virginica'}

from matplotlib import pyplot as plt
import numpy as np
import math


label_dict = {1: 'Iris-Setosa',
              2: 'Iris-Versicolor',
              3: 'Iris-Virgnica'}

feature_dict = {0: 'sepal length [cm]',
                1: 'sepal width [cm]',
                2: 'petal length [cm]',
                3: 'petal width [cm]'}




from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)

# Shortcut - PCA in scikit-learn

from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=2)
Y_sklearn = sklearn_pca.fit_transform(X_std)

def plot_pca():

    ax = plt.subplot(111)

    for label,marker,color in zip(
        range(1,4),('^', 's', 'o'),('blue', 'red', 'green')):

        plt.scatter(x=Y_sklearn[:,0][y == label],
                y=Y_sklearn[:,1][y == label],
                marker=marker,
                color=color,
                alpha=0.5,
                label=label_dict[label]
                )

    plt.xlabel('PC1')
    plt.ylabel('PC2')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title('PCA: Iris projection onto the first 2 principal components')

    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
            labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)    

    plt.tight_layout
    plt.grid()

    plt.show()
    
plot_pca()

