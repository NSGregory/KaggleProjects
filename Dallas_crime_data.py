
# ****************************************************************************
# Libraries
# ****************************************************************************
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ****************************************************************************
# End Libraries
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

#Dallas police incident data
DPI_data = pd.read_csv("Police_Incidents.csv")
#DPI_data_index = pd.read_csv("Police_Incidents.csv", index_col="Year of Incident")

#Establish the unique string data across important categories
problem_types = DPI_data["Call (911) Problem"].unique().tolist()
incident_types = DPI_data["Type of Incident"].unique().tolist()
property_types = DPI_data["Type of Property"].unique().tolist()
location_types = DPI_data["Type  Location"].unique().tolist()

#find the years contained within the data
data_years = DPI_data["Year of Incident"].unique().tolist()
removal_list = [2109, 2048, 2013, 2009, 2005] #by analaysis of data
for x in removal_list:
    data_years.remove(x)

#data_years.remove([2109,2048,2013,2009,2005]) #manual investigation found an incorrect year, removed

#break down by years
DPI_2017 = DPI_data[DPI_data["Year of Incident"] == 2017]
crime_counts_2017 = DPI_2017['Call (911) Problem'].value_counts()
#sns.barplot(x=crime_counts_2017, y=crime_counts_2017.index)

# for year in data_years:
#     isolated_year_data = DPI_data[DPI_data['Year of Incident'] == year]
#     crime_counts = isolated_year_data['Call (911) Problem'].value_counts()
#     sns.kdeplot(data=crime_counts, label=year, shade=True)

#plt.show()

#Concept for pulling data together across years to create a new dataframe
def axis_count_by_year(dataFrame, axis_of_interest):
    """Looks at the Police_incident.csv dataset and produces a dataframe that
    groups the data of interest by year, identifies the unique entries of the
    data of interest, and provides a count of that item.

    e.g. It will pull out a list of all crimes reported and then count the
    number of times each is reported then display it by year in a dataframe.

    dataFrame : the source dataframe to pull the data from
    axis_of_interest : the column within dataFrame to be evaluated

    returns data sorted by year in a dataframe
    """
    index_entries = DPI_data[axis_of_interest].unique().tolist()

    data_years = sorted(dataFrame["Year of Incident"].unique().tolist())
    removal_list = [2109, 2048, 2013, 2009, 2005] #by analaysis of data
    for x in removal_list:
        data_years.remove(x)

    tmp = pd.DataFrame(columns=data_years, index=index_entries)
    for year in data_years:
        isolated_year = dataFrame[dataFrame['Year of Incident'] == year]
        crime_counts = isolated_year[axis_of_interest].value_counts()
        tmp[year] = crime_counts
    return tmp

#parameters for graphing a very large heatmap -- only legible as the png
#which you can zoom in on and scroll through
#looks bad via plt.show()
# fig, ax = plt.subplots(figsize=(30,25))
# hm = sns.heatmap(data=tmp, cbar=True, annot=True, cmap=plt.cm.Reds)
# plt.tight_layout()
# plt.savefig("Crime_In_Dallas.png", dp=400)
