import click
import pandas as pd
import numpy as np
from apriori import apriori
from preprocess import preprocess
from kmeans import kmeans
from dtree import dtree 
from naivebayes import naivebayes
    

@click.group()
def root():
    "Data Mining Project"

# root.add_command(hello)
root.add_command(apriori)
root.add_command(preprocess)
root.add_command(kmeans)
root.add_command(dtree)
root.add_command(naivebayes)

if __name__ == "__main__":
    root()
