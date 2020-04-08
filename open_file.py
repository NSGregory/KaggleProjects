# ****************************************************************************
# example for how to pull data from KaggleData Folder
#
# This assumes the folder containing your data is in the same starting directory
# as the script you are using.
#
# If you need to go up a level in the tree, add ../ to the file path for each
# level that you need to go up.
#
# If you want to specify the entire filepath, use r"C:\file\path\filename.csv"
# The r in front of the filepath is intention.  This is for windows.
#
# ****************************************************************************


import pandas as pd
data_folder = "KaggleData/"
filename = "cancer_b.csv"
filepath = data_folder+filename
dataset = pd.read_csv(filepath)





