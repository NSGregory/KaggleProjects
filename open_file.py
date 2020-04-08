# ****************************************************************************
# example for how to pull data from KaggleData Folder
#
# This assumes the folder containing your data is in the same starting directory
# as the script you are using.
#
# ****************************************************************************


import pandas as pd
data_folder = "KaggleData/"
filename = "cancer_b.csv"
filepath = data_folder+filename
dataset = pd.read_csv(filepath)





