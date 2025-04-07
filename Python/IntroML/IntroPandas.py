# %%
"""
#Pandas

You will now learn about Pandas and how to load and play with real data.

**Pandas** is a Python package that is designed to make it easy to work with data, and has many useful functions for loading data from files, manipulating and displaying data, and storing data to files.

Let's first import Pandas and Numpy.
"""

# %%
import numpy as np
import pandas as pd

# %%
"""
###Loading some sample data

Let's first see what files are already included in Google Colab. We can use the 'ls' command to list the current directory and see what files and folders we have.


"""

# %%
%%bash
ls -l

# %%
"""
Apparently Google Colabs provides some sample data sets in the folder 'sample_data'. Let's list them with the ls command.
"""

# %%
%%bash
ls -lh sample_data

# %%
"""
We see some datasets stored in Comma Separated Value (csv) files and a json file, as well as a Readme file.

Some notes on 'ls' (which you can ignore):

1. The -l flag asks for information to be listed for each file, including file size. The -h flag asks for the file sizes to be in MB, KB (human-readable), instead of bytes.

2. The exclamation mark ! we keep using is an 'escape character' that allows us to run commands outside of Python in the shell (Google Colabs runs a Bash shell, like in Unix operating systems). We won't use this very often.

Let's view the readme file to see about the datasets. Here, we use the shell command 'cat'
"""

# %%
%%bash
cat sample_data/README.md

# %%
"""
Let's open the california housing data, which is stored as a csv file. A csv file is just a text file with the data stored as plain text separated by commas. A lot of our datasets will be stored this way.

To view a bit of the csv file, we can use the 'head' shell command.
"""

# %%
%%bash
head -n 20 sample_data/california_housing_train.csv

# %%
"""
It would be rather complicated to write a Python function to read a csv file and store the data in a Python list or Numpy array.

This is where Pandas is extremely useful. Pandas has built in functions for reading csv files (and many other data types). We can read the data with Pandas using the code below.
"""

# %%
data_train = pd.read_csv('sample_data/california_housing_train.csv')
data_test = pd.read_csv('sample_data/california_housing_test.csv')

# %%
"""
We ran the code and it appears nothing happend! The code above read the csv files and stored their data in the 'data_train' and 'data_test' variables. We didn't ask it to produce any ouptut.

We can now use Pandas to look at the data and try to understand it better. Let's check the size first.
"""

# %%
data_train.shape

# %%
data_test.shape

# %%
"""
So the training data is a Data Frame (an array) with 17,000 data points, each of which has 9 features. The testing data has 3,000 data points.

But what does this data look like? Let's look at the first few rows.
"""

# %%
data_train.head()

# %%
"""
It seems each row corresponds to a county in California, and the row includes various statistics of that county.
"""

# %%
"""
If you just want to know the column headers.
"""

# %%
data_train.columns

# %%
"""
Pandas also has a 'describe' function that provides a useful summary of your data.
"""

# %%
data_train.describe()

# %%
"""
##Downloading data from the internet
Another great place to find datasets is [Kaggle](https://www.kaggle.com/datasets) and [Github](https://www.github.com). If you have a URL, pandas can read a csv file directly from the internet, or you can download it first and load a local file. Let's download some covid infection data from the Nytimes [Covid data repository](https://github.com/nytimes/covid-19-data).
"""

# %%
import pandas as pd
covid_data = pd.read_csv('https://github.com/nytimes/covid-19-data/raw/master/us-counties-2020.csv')
covid_data.head(n=50000)

# %%
covid_data.describe()

# %%
"""
Sometimes, when downloading a dataset, we do not get a URL we can use directly. This is the case with [Kaggle](https://www.kaggle.com/datasets). In this case, we can import the file by clicking on the folder on the left and clicking the upload button (the page with the up arrow). Then select the file from your hard drive to upload.

"""

# %%
"""
##Exercise 1
1. Find another dataset as a csv file on Github or Kaggle. For example, try another csv file from the Nytimes [Covid data repository](https://github.com/nytimes/covid-19-data).

2. Read the csv file into Colab and use the Pandas 'head()' and 'describe()' functions to view the data.
"""

# %%
#Put your code here, or make more code boxes
df = pd.read_json('sample_data/anscombe.json')
df

# %%
"""
#Manipulating and visualizing data
Pandas has lots of useful functions for manipulating data. Let's go back and use our California housing data.
"""

# %%
data_train = pd.read_csv('sample_data/california_housing_train.csv')

# %%
"""


First, we can sort the data by any of the features. For example
"""

# %%
data_train.sort_values(by='median_income', ascending=True)

# %%
"""
Suppose you just want to focus on just one column of the dataset. We can extract a column by label.
"""

# %%
age = data_train['housing_median_age']
print(age)

# %%
"""
We can now compute mean, median, or standard devaition easily.
"""

# %%
age.mean()

# %%
age.median()

# %%
age.std()

# %%
"""
We can also visualize data by creating a histogram.
"""

# %%
age.hist(bins=50)

# %%
"""
Question: Why do we see a spike at 52 years old? You may want to look at the website for the dataset, from the readme file above.
"""

# %%
"""
##Exercise 2

1. Figure out why there is a spike at 52 years old in the data histogram above.
2. Load another dataset, as in the last exercise, and sort it by different columns. Create histograms of some of the columns. What do you see?
"""

# %%
#Put your code here, or make another code box.

# %%
"""
##Subsetting data

Sometimes we wish to subset our data; that is, remove some columns or rows that are not needed, or separate data from labels. This is easy with Pandas.

To get the first 5 rows:
"""

# %%
data_train[0:20]

# %%
"""
To select only some columns (columns 1,2,4,7) and the first 5 rows.
"""

# %%
data_train.iloc[0:5,[1,2,4,7]]

# %%
"""
To do the same thing but select all rows:
"""

# %%
data_train.iloc[:,[1,2,4,7]]

# %%
"""
You can also use the labels of the columns, instead of their numbers. Notice we used 'iloc' above and 'loc' below (the i is for index).
"""

# %%
data_train.loc[:,['latitude','housing_median_age','total_bedrooms','median_income']]

# %%
"""
You can index with any Python list or using notation 'a:b' to specify the range a,a+1,...,b-1. If a=0 it can be omitted (same if b is the length of the array). You can use any combination of these to select rows and columns from your data. A colon : by itself means all rows or columns.
"""

# %%
"""
You can also select data matching some criteria. Say you want to look at counties with median age at least 20.
"""

# %%
I = data_train['housing_median_age'] > 20
print(I)
data_train[I]

# %%
"""
In the code above, I is a boolean array of True or False, based on whether the row satisfies housing_median_age > 20. The line data_train[I] is called *logical indexing*. It allows indexing Pandas data frames (or Numpy arrays) by Boolean (or logical) arrays.

You can combine these with logical operations. Say we also want to subset the data to median_income < 2. The code below creates another boolean array J indicating when median income is less than 2, and then uses the logical 'and' with 'I & J' to index where both I and J are true.
"""

# %%
J = data_train['median_income'] < 2
data_train[I & J]  #Shows data where median_income < 2 and housing_median_age > 20
data_train[I | J]  #Shows data where median_income < 2 or housing_median_age > 20

# %%
"""
To subset the data where median_income < 2 or housing_median_age > 20, use the logical or 'I | J'.
"""

# %%
data_train[I | J]  #median_income < 2 or housing_median_age > 20

# %%
"""
Finally, you may also need to change values of your data (say, they are corrupted).
"""

# %%
data_train['housing_median_age'][0:314] = range(314)
data_train.head(n=20)

# %%
"""
Can you spot which value was changed above?
"""

# %%
"""
There are lots of other useful functions in Pandas. Please see [Pandas Tutorial](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html) for more details.
"""

# %%
"""
##Saving datasets to files and to your computer
Now that we have changed our dataset, we may wish to save it to a csv file.
"""

# %%
data_train.to_csv('test.zip',compression='infer')

# %%
"""
To check the file has been created, list the directory.
"""

# %%
!ls

# %%
"""
If you now want to download the modified file to your computer:
"""

# %%
from google.colab import files
files.download('test.zip')

# %%
"""
This can also be used for uploading files.
"""

# %%
files.upload()

# %%
"""
##Exercise 3

Use the dataset you downloaded in the earlier exercise.

1. Subset the data in different ways, using basic indexing and some logical indexing.
2. Modify the data and save it in a new csv file.
3. Download the csv file to your computer.
"""

# %%
#Code here

# %%
"""
##Challenge Exercise

1. Find a dataset that interests you on [Kaggle](https://www.kaggle.com/datasets).
2. Download the csv file for the dataset to your computer and upload it to Google Colab.
3. Load it into Pandas and view the data, sort by various columns, subset the data in different ways, and use the 'describe()' function.


"""

# %%
#Code here
