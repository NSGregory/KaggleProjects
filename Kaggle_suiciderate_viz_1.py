# ****************************************************************************
#
# These are notes and code snippets taken from Kaggle's
# Data Visualization: from Non-coder to Coder
#
# This course utilizes the seaborn library.
#
# ****************************************************************************

# ****************************************************************************
#
#   This section is for the basic setup and libraries used in the course
#
# ****************************************************************************
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# ****************************************************************************
#
# End library call
#
# ****************************************************************************

# ****************************************************************************
#           Basic Pandas function: pull data out of the CSV file
#	 Dependency: pandas
# ****************************************************************************

import pandas as pd
filename = "suicide_rate.csv"
#raw data from csv
dataset = pd.read_csv(filename)

#this uses the "year" column as the index and to treat those values as dates
df_dates = pd.read_csv(filename, index_col="year", parse_dates=True)
df_plot_countries = pd.read_csv(filename, index_col="year", parse_dates=True)


#we need to do some cleaning of the data to facilitate analysis
#first we convert 'male' and 'female' to 1 and 0
df_dates['sex'].replace({'female':0, 'male':1}, inplace=True)

#this converts generations to values from 0 to 5
df_dates['generation'].replace(list(set(df_dates['generation'])),
                               list(range(len(set(df_dates['generation'])))),
                               inplace = True)
#change country to numbers
df_dates['country'].replace(list(set(df_dates['country'])),
                                list(range(len(set(df_dates['country'])))),
                                inplace = True)

#GDP is a string in format XXX,XXX,XXX  -- need to remove commas

#attempted breaking up the lines to speed it up, no benefit over single line
#GDP_x = list(df_dates[' gdp_for_year ($) '])
#GDP_y = [float(n.replace(',','')) for n in df_dates[' gdp_for_year ($) ']]
#df_dates[' gdp_for_year ($) '].replace(GDP_x, GDP_y, inplace = True)


#this works to convert a series numeric series with commas into a float but it
#is slow and the scale of this data outstrips the rest of the values
# df_dates[' gdp_for_year ($) '].replace(list(df_dates[' gdp_for_year ($) ']),
#                                             [float(n.replace(',','')) for
#                                             n in df_dates[' gdp_for_year ($) ']],
#                                             inplace = True
#                                             )
#this is much faster than the above solutions
df_dates[' gdp_for_year ($) '] = df_dates[' gdp_for_year ($) '].apply(
                                            lambda x: float(x.replace(',','')))


#this drops the country-year column because its hard to deal with and i honestly
#don't understand what it means; the age is difficult because it is a range
df_dates.drop(['age','country-year'], axis=1, inplace=True)

#get rid of any rows missing data -- not a great idea, but just trying to get
#a basic look at the data
df_dates.dropna(axis='index', how='any', inplace=True)
print(df_dates.head())
# ****************************************************************************

# ****************************************************************************
# Basic Plotting
# ****************************************************************************

# Create a separate dataframe for plotting purposes
df_plot = df_dates
#this reduces the scale for graphing purposes
# df_plot[' gdp_for_year ($) '] = df_plot[' gdp_for_year ($) '].apply(
#                                                            lambda x: x/(10**9))
# df_plot['population'] = df_plot['population'].apply(lambda x: x/(10**6))
# df_plot['gdp_per_capita ($)'] = df_plot['gdp_per_capita ($)'].apply(
#                                                             lambda x: x/(1000))

dash_styles = ["",
               (4, 1.5),
               (1, 1),
               (3, 1, 1.5, 1),
               (5, 1, 1, 1),
               (5, 1, 2, 1, 2, 1),
               (2, 2, 3, 1.5),
               (1, 2.5, 3, 1.2),
               (3,3,3,3)]

# Line chart; looking at minimal data here since we will have to sort out
# the many values contained in the dataset to represent them more clearly
#sns.lineplot(data=df_plot, hue='country', style='country', dashes=dash_styles)

#plt.show()
# ****************************************************************************
# End Basic Plotting
# ****************************************************************************

# ****************************************************************************
#
#            Splitting training and testing
#
# ****************************************************************************
# We need to divide up the data into a training set and a testing set, because
# if we test the model on the data we train it with, we will get "perfect"
# results that don't necessarily indicate how good the model actually is.

from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
#
# train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
#
# requires X and y be defined, which typically occurs while you are building the
# the predictive model.
#
# Dependency: sklearn
# ****************************************************************************
#       End: Splitting Training and Testing
# ****************************************************************************

# ****************************************************************************
#
#           This Section for How to Define a Model Using
#                 Random Forest Regressor
#
# ****************************************************************************
# This section establishes the model.  Features are the data within the model
# that will be used to predict the target value.  By convention it is labeled X.
# X will be a list of Series from within the original dataset.
#
#        X = original_dataset[List_Of_Column_Titles]
#
# The Target is the value we ultimately wish to predict.  When building the
# model, the target is a series.  By convention it is labeled y.
#
#        y = original_dataset.target
#
#	Dependency: sklearn
#
# ****************************************************************************
from sklearn.ensemble import RandomForestRegressor

dataset_features = ['country', 'sex', 'population', 'HDI for year',
                    ' gdp_for_year ($) ', 'gdp_per_capita ($)', 'generation']
y = df_dates.suicides_no
X = df_dates[dataset_features]

# splits y and X into training and testing data
# (see Splitting Training and Testing: ms-split_train)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

evaluated_model = RandomForestRegressor(random_state=5)
evaluated_model.fit(train_X, train_y)

#show the first few lines of the dataset
print(evaluated_model.predict(val_X.head()))

# ****************************************************************************
#       End Building the DecisionTreeRegressor
# ****************************************************************************

# Sklearn:
# ml-split_train
# ml-mae
# ml-decision_tree_regressor
# ml-random_forest_regressor

# ****************************************************************************
#
#                   Mean Absolute Error
#
#   A tool for assessing how good the model is at predicting the correct value.
#
# For each entry in the dataframe, error can be defined as:
#
#          error = actual - predicted
#
# Mean absolute error is calculated by taking the absolute value of error for
# each of the predicted values generated by the model in the dataframe and then
# calculating the mean of those values.
#
#	Dependency: sklearn
# ****************************************************************************

from sklearn.metrics import mean_absolute_error

# ****************************************************************************

prediction = evaluated_model.predict(val_X)
error = mean_absolute_error(val_y, prediction)
print(error)

# ****************************************************************************
# End Mean Absolute Error
# ****************************************************************************
