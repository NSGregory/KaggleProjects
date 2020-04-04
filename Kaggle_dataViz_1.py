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
filename = "spotify.csv"
#raw data from csv
dataset = pd.read_csv(filename)

#this uses the "year" column as the index and to treat those values as dates
spotify_data = pd.read_csv(filename, index_col="Date", parse_dates=True)
# ****************************************************************************
# End CSV_read
# ****************************************************************************

# ****************************************************************************
# Basic Seaborn Lineplot
# ****************************************************************************
# plt.figure(figsize=(14,6))
# plt.title("Daily Global Streamms of Popular Songs")
# sns.lineplot(data=spotify_data)
#plt.show()
# ****************************************************************************
# End Seaborn Lineplot
# ****************************************************************************

filename = "museum_visitors.csv"
museum_data = pd.read_csv(filename, index_col="Date", parse_dates=True)
#sns.lineplot(data=museum_data)

filename = "flight_delays.csv"
flight_data = pd.read_csv(filename, index_col="Month")

#make a seaborn bar graph
# plt.figure(figsize=(10,6))
# plt.title("Average Arrival Delay for Spirit Airlines Flights by Month")
# sns.barplot(x=flight_data.index, y=flight_data['NK'])
# plt.ylabel("Arrival delay (in minutes)")
#
# #make a seaborn heatmap
# plt.figure(figsize=(16,8))
# plt.title("Average Arrival Delay for Each Airline by Month")
# sns.heatmap(data=flight_data, annot=True)
# plt.xlabel("Airline")

# ****************************************************************************
# Bar chart and heatmap exercise
# ****************************************************************************
ign_data = pd.read_csv("ign_scores.csv", index_col="Platform")
# sns.barplot(x=ign_data.index, y=ign_data["Action"])
# plt.show()

pc=ign_data[ign_data.index == "PC"]
highest_PC_score = pc.max(axis='columns')[0]

psv=ign_data[ign_data.index == "PlayStation Vita"]
#print (psv.min(axis='columns'))
psv_min_cat = psv.idxmin(axis='columns')[0]

# sns.barplot(y=ign_data.index, x=ign_data['Racing'])
# plt.show()

# sns.heatmap(data=ign_data, annot=True)
# best_score = ign_data.max(axis='columns').max(axis='index')
# best_system = ign_data.max(axis='columns').idxmax(axis='index')
# best_genre = ign_data.max(axis='index').idxmax(axis='index')


# ****************************************************************************
#
#  Scatterplots
#
# ****************************************************************************

# insurance_data = pd.read_csv("insurance.csv")
# sns.scatterplot(x=insurance_data['bmi'],
#                 y=insurance_data['charges'],
#                 hue=insurance_data['smoker'])
# sns.regplot(x=insurance_data['bmi'], y=insurance_data['charges'])
#
#
# sns.lmplot(x='bmi', y='charges', hue='smoker', data=insurance_data)
#
#
# sns.swarmplot(x=insurance_data['smoker'],
#               y=insurance_data['charges'])

# ****************************************************************************
# End of Scatterplot
# ****************************************************************************


# ****************************************************************************
#
# Scatterplot Exercise
#
# ****************************************************************************

# candy_data = pd.read_csv("candy.csv")
# # sns.scatterplot(x=candy_data['sugarpercent'],
# #                 y=candy_data['winpercent']
# #                 )
# #
# # sns.regplot(x=candy_data['sugarpercent'],
# #             y=candy_data['winpercent']
# #             )
# #
# # sns.scatterplot(x=candy_data['pricepercent'],
# #                 y=candy_data['winpercent'],
# #                 hue=candy_data['chocolate']
# #                 )
#
# #sns.lmplot(x='pricepercent', y='winpercent', hue='chocolate', data=candy_data)
#
# # sns.swarmplot(x=candy_data['chocolate'],
# #               y=candy_data['winpercent']
# #              )


# ****************************************************************************
#
# Histograms and Density Plots: Notes
#
# ****************************************************************************

iris_data = pd.read_csv("iris.csv")
iris_ver_data = pd.read_csv('iris_versicolor.csv')
iris_set_data = pd.read_csv('iris_setosa.csv')
iris_vir_data = pd.read_csv('iris_virginica.csv')

#print (iris_data.head())

# a = the column you want to plot
# kde = kernal density estimate
#       changing this to true gives a slightly different graph

#sns.distplot(a=iris_data['Petal Length (cm)'], kde=False)
#sns.distplot(a=iris_data['Petal Length (cm)'], kde=True)

#sns.kdeplot(data=iris_data['Petal Length (cm)'], shade=True)

# sns.jointplot(x=iris_data['Petal Length (cm)'],
#              y=iris_data['Sepal Width (cm)'],
#              kind='kde')

# sns.distplot(a=iris_set_data['Petal Length (cm)'], label="Iris setosa", kde=False)
# sns.distplot(a=iris_ver_data['Petal Length (cm)'], label="Iris versicolor", kde=False)
# sns.distplot(a=iris_vir_data['Petal Length (cm)'], label="Iris virginica", kde=False)
# plt.title("Histogram of Petal Lengths by Species")
# plt.legend()

# sns.kdeplot(data=iris_set_data['Petal Length (cm)'], label="Iris setosa", shade=True)
# sns.kdeplot(data=iris_ver_data['Petal Length (cm)'], label="Iris versicolor", shade=True)
# sns.kdeplot(data=iris_vir_data['Petal Length (cm)'], label="Iris virginica", shade=True)
# plt.title("Distribution of Petal Lengths by Species")
# plt.legend()
#
# plt.show()

# ****************************************************************************
# End notes: Histograms and density plots
# ****************************************************************************

# ****************************************************************************
#
# Histograms and Density Plots Exercises
#
# ****************************************************************************
