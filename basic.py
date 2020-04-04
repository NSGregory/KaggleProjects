#
#
#  Learning Basic Functions for Kaggle
#
#

import pandas as pd
from sklearn.model_selection import train_test_split


filename = "undefined" #insert the name of the file here
dataset = pd.read_csv(filename)

dataset_features = [] #insert features of the dataset in '' separated by ,
y = dataset.target
X = dataset[dataset_features]

# splits y and X into training and testing data
# (see above: Splitting Training and Testing)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)


