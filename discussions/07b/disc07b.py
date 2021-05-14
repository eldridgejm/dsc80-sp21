import pandas as pd
import numpy as np
import os
import re

# Q1

def duplicate_words(s):
    """
    Provide a list of all words that are duplicates in an input sentence.
    Assume that the sentences are lower case.

    :Example:
    >>> duplicate_words('let us plan for a horror movie movie this weekend')
    ['movie']
    >>> duplicate_words('I like surfing')
    []
    >>> duplicate_words('the class class is good but the tests tests are hard')
    ['class', 'tests']
    """

    return ...


# Q2
def laptop_details(df):
    """
    Given a df with product description - Return df with added columns of 
    processor (i3, i5), generation (9th Gen, 10th Gen), 
    storage (512 GB SSD, 1 TB HDD), display_in_inch (15.6 inch, 14 inch)

    :Example:
    >>> df = pd.read_csv('data/laptop_details.csv')
    >>> new_df = laptop_details(df)
    >>> new_df.shape
    (21, 5)
    >>> new_df['processor'].nunique()
    3
    """

    return ...


# Q3
def corpus_idf(corpus):
    """
    Given a text corpus as Series, return a dictionary with keys as words 
    and values as IDF values

    :Example:
    >>> reviews_df = pd.read_csv('data/musical_instruments_reviews.csv')
    >>> idf_dict = corpus_idf(reviews_df['reviewText'])
    >>> isinstance(idf_dict, dict)
    True
    >>> len(idf_dict.keys())
    2085
    
    """

    return ...
