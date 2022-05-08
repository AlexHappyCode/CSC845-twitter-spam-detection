# Hi, This is Alex's random forest model
#### This model classifies tweets as spam or ham

# Setting Up The Notebook

The notebook that we want to run to showcase the algorithm is
random\_forest.ipynb. 
To do this we have to have the following packages installed in 
your python or anaconda environment:

- pandas
- numpy
- functools
- matplotlib
- nltk

This notebok will also download a package that includes common stem words and stop words
with the following two lines in order to run in your home directory.

nltk.download('stopwords')
nltk.download('punkt')

# Running the Notebook

Once the packages are installed, simply run all cells and scroll
down to see the results and the number of mismatches displayed
in the pie chart. The final pie chart in the notebook showcases
the random forest after having been run on the entire dataset

