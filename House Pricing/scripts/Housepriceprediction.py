The general process to get from data to predictive model tends to involve three major components:

Getting to know the data.
Cleaning and preparing the data for modelling.
Fitting models and evaluating their performance.

'''
# Section 1: Getting to know the data. 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data_train = pd.read_csv('E:/ds_practice/case_study/housePricing/dataset/train.csv')
data_test = pd.read_csv('E:/ds_practice/case_study/housePricing/dataset/test.csv')
data_train.info()

# ome features are inherently numerical - total living area (GrLivArea)
# Some features are inherently discrete - the number of rooms (TotRmsAbvGrd).
# Other features are categorical - the neighbourhood in which the house is located (Neighborhood)
# the type of foundation the house was built on (Foundation)
# some of ordinal: the overall quality rating (OverallQual) or the irregularity of the lot (LotShape)

# EXPLORE AND VISUALIZE THE DATA


'''
The distribution of the target variable and of individual features (univariate analysis).
The relationship between pairs of variables (bivariate analysis).

'''

#  1. Univariate analysis---------------------------------------------

# for Numerical variables
# Let’s begin by generating one for SalePrice, our target variable.

#plt.hist(data_train.SalePrice, bins = 25)
# To make the distribution more symmetric, we can try taking its logarithm:
#plt.hist(np.log(data_train.SalePrice), bins = 25)

# we can think of log(SalePrice) as our true target variable.


# Categorical variables

# For categorical variables, bar charts and frequency counts are the natural alternative to histograms

#data_train.Foundation.value_counts()
#sns.countplot(data_train.Foundation)

# Here we can immediately see that only two types of 
# foundation (poured concrete (PConc) and cinderblock (CBlock)) dominate in our dataset.

# This might think  to make a transformation.
# For example, depending on the type of model we decide to use, we may want to merge Stone, Wood and Slab into a single (‘other’) category.

# Alternatively, if we think stone and wood houses are very important, it may alert us to the fact that we have a deficiency in our data, and need to go out and collect more stone and wood examples.
# Related to this last point, it is also important to check whether the distributions in our training and test sets are similar to each other.

#sns.countplot(data_test.Foundation)

# Bivariate analysis-----------------------------------------------

'''
most interesting will be the relationship between the 
target variable (sale price) and the features we will 
use for prediction. However, as we will see, studying 
relationships among features can also be important.

'''

# Numerical variables

# For numerical features, scatter plots are the go-to tool.

plt.plot(data_train.GrLivArea, data_train.SalePrice,
         '.', alpha = 0.3)
 
plt.plot(data_train.GrLivArea, np.log(data_train.SalePrice),
         '.', alpha = 0.3)


plt.plot(data_train.YearBuilt, data_train.GarageYrBlt,
         '.', alpha=0.5, label = 'training set')
 
plt.plot(data_test.YearBuilt, data_test.GarageYrBlt,
         '.', alpha=0.5, label = 'test set')
 
plt.legend()


# we might consider creating a new feature that tells us whether or not a garage was originally constructed with the house or how many years later one was added.


# Categorical variables
# http://seaborn.pydata.org/tutorial/categorical.html
# stripplot, pointplot, boxplot and violinplot
sns.stripplot(x = data_train.Neighborhood.values, y = data_train.SalePrice.values,
              order = np.sort(data_train.Neighborhood.unique()),
              jitter=0.1, alpha=0.5)
 
plt.xticks(rotation=45)


Neighborhood_meanSalePrice = data_train.groupby('Neighborhood')['SalePrice'].mean()
 
Neighborhood_meanSalePrice = Neighborhood_meanSalePrice.sort_values()

sns.pointplot(x = data_train.Neighborhood.values, y = data_train.SalePrice.values,
              order = Neighborhood_meanSalePrice.index)
 
plt.xticks(rotation=45)


def count_missing(data):
    null_cols = data.columns[data.isnull().any(axis=0)]
    X_null = data[null_cols].isnull().sum()
    X_null = X_null.sort_values(ascending=False)
    print(X_null)
 
data_X = pd.concat([data_train.drop('SalePrice',1), data_test])
count_missing(data_X)

# Missing for a reason

# some of the missing values are in fact meaningful. For example, missing values for garage, pool or basement-related features simply imply that the house does not have a garage, pool or basement respectively.
# In this case, it makes sense to fill these missing values with something that captures this information.

# we can replace missing values in such cases with a new value called ‘None’
catfeats_fillnaNone = ['Alley',
    'BsmtCond','BsmtQual','BsmtExposure',
    'BsmtFinType1', 'BsmtFinType2',
    'FireplaceQu',
    'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'PoolQC',
    'Fence',
    'MiscFeature']
 
data_X.loc[:,catfeats_fillnaNone] = data_X[catfeats_fillnaNone].fillna('None')

# for most numerical features of this kind, it makes sense to replace the missing values with zero:

numfeats_fillnazero = ['BsmtFullBath', 'BsmtHalfBath', 'TotalBsmtSF',
     'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
     'GarageArea', 'GarageCars']
 
data_X.loc[:,numfeats_fillnazero] = data_X[numfeats_fillnazero].fillna(0)

data_X.loc[:,'GarageYrBlt'] = data_X['GarageYrBlt'].fillna(data_X.YearBuilt)

count_missing(data_X)

# delete, impute or leave.

# 1. Delete

'''

In cases where a very large portion of values are missing for a given feature, 
we may simply decide to drop that feature (column) altogether. 
Similarly, if almost all feature values are missing for a given entry (row),
we may decide to delete that row. 

The downside, of course, 
is losing potentially valuable information, which is particularly problematic
if the dataset is small compared to the number of rows or columns 
with missing values. In the house prices dataset, the fraction of missing
entries in any given row or column is not very high 
(at most 17% missing for LotFrontage), and it will likely be better 
to keep all of our data.
'''

# 2. Impute

'''
If we do decide to keep all of our data, we will generally need to fill 
in – or ‘impute’ – the missing entries. The majority of machine learning 
algorithms cannot handle null values, so modelling will only be possible 
after we do so.

The crudest option is to simply replace each missing entry by the mean, 
median or mode of the given feature, which gives us the roughest possible 
estimate for what the missing value might be. We can implement this for 
our house prices dataset as follows (using mode and median for categorical 
and numerical features respectively)


A more sophisticated approach might involve using what we know about the 
feature’s relationship to other features to guess the missing values.
 For example, if we have 5 features (F1, … F5) and F1 has some missing 
 values, we can treat F1 as our target variable and train a model on 
 (F2, … F5) to predict what the missing values might be.

'''


catfeats_fillnamode = ['Electrical', 'MasVnrType', 'MSZoning', 'Functional', 'Utilities',
     'Exterior1st', 'Exterior2nd', 'KitchenQual', 'SaleType']
 
data_X.loc[:, catfeats_fillnamode] = data_X[catfeats_fillnamode].fillna(data_X[catfeats_fillnamode].mode().iloc[0])
 
numfeats_fillnamedian = ['MasVnrArea', 'LotFrontage']
 
data_X.loc[:, numfeats_fillnamedian] = data_X[numfeats_fillnamedian].fillna(data_X[numfeats_fillnamedian].median())

# 3. Leave

'''
While the majority of machine learning algorithms cannot handle missing 
values natively, exceptions do exist. For example, the powerful and 
widely used xgboost library happily handles data of this kind. 
'''
# NON-NUMERICAL FEATURES

print(data_X.dtypes.value_counts())

print(data_X.select_dtypes(include = [object]).columns)

'''
Since the majority of available machine learning algorithms can only take numbers 
(floats or integers) as inputs, we must encode these features numerically 
if we are to use them in our models.

non-numerical variables tend to come in two flavours: ordinal and categorical. 
Ordinal variables – such as OverallQual or LotShape – have an intrinsic order to them, 
while purely categorical variables – such as Neighborhood or Foundation
'''

# Ordinal features

'''
Since ordinal features are inherently ordered, they lend themselves 
naturally to numerical encoding. For example, the possible values for 
LotShape are Reg (regular), IR1 (slightly irregular), IR2 (moderately irregular)
and IR3 (irregular), to which we could assign the values (0,1,2,3) respectively:

'''

data_X.LotShape = data_X.LotShape.replace({'Reg':0, 'IR1':1, 'IR2':2, 'IR3':3})

'''
This is known as ordinal encoding and is the most straightforward approach for 
encoding non-numerical variables: we simply assign a number to each possible 
value a feature can take. Mapping the levels of an ordinal variable to 
consecutive integers, as we have done above, is good in that it keeps 
the relative relationship between values intact.

'''

# Categorical features

# Ordinal encoding

print(data_test.Neighborhood.head(15))
print(pd.Series(pd.factorize(data_test.Neighborhood.head(15))[0]))

# Dummy encoding (aka one-hot encoding)

print(pd.get_dummies(data_test.Neighborhood.head(15), drop_first=True))

'''
The great advantage of dummy encoding is that it doesn’t impose any ordering 
on our data and ensures that the distance between each pair of values 
(neighbourhoods in this case) is the same.
'''



print(data_X.columns)

from sklearn import linear_model

X = data_train['GrLivArea'].values
m = len(X)
X = X.reshape((m, 1))
y = np.log(data_train.SalePrice)
y_test = data_test['GrLivArea'].values
n = len(y_test)
y_test = y_test.reshape((n, 1))
ols = linear_model.LinearRegression()
ols.fit(X, y)
y_test_predicted = ols.predict(y_test)
print(y_test_predicted)

y_test_predicted_dollars = np.exp(y_test_predicted)
print(y_test_predicted_dollars)

# How well does model work ?
# EVALUATION METRIC
# In-sample vs out-of-sample error
# Validation and cross validation
# Define training set and Validation set 
# k-fold cross validation.
from sklearn.model_selection import cross_val_score
 
scores = cross_val_score(ols, X, y, cv=5,
                         scoring = 'neg_mean_squared_error')
 
scores = np.sqrt(abs(scores))
 
print("CV score: ", scores.mean())

from sklearn.model_selection import train_test_split
 
X_test, X_train, y_test, y_train = train_test_split(X, y, test_size=0.8, shuffle = True)
 
ols.fit(X_train, y_train)
y_test_predicted = ols.predict(X_test)
dollar_errors = np.exp(y_test) - np.exp(y_test_predicted)
print(dollar_errors)
percentage_errors = dollar_errors/np.exp(y_test) * 100
print(percentage_errors)

plt.hist(dollar_errors, bins = np.linspace(-140000, 140000, 40))
plt.xlabel('$ error in sale price')
 
plt.hist(percentage_errors, bins = np.linspace(-140,140,50))
plt.xlabel('% error in sale price')

















































