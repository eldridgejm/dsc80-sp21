import re
import pandas as pd
import os

# Question 1

def merge_majors(df1, df2):
    '''
    Merge the two input dataframes on major code number
    >>> df1 = pd.read_csv('data/majors-list.csv')
    >>> df2 = pd.read_csv('data/majors-data.csv')
    >>> merged = merge_majors(df1, df2)
    >>> len(merged) == len(df1)
    True
    >>> len(merged.columns)
    10
    >>> 'FOD1P' in merged.columns
    False
    '''
    return pd.merge(df1, df2, left_on='FOD1P', right_on='Major_code', how='left').drop('FOD1P', axis=1)


def best_majors(df):
    '''
    Return a list of "best" majors
    >>> df1 = pd.read_csv('data/majors-list.csv')
    >>> df2 = pd.read_csv('data/majors-data.csv')
    >>> merged = merge_majors(df1, df2)
    >>> best = best_majors(merged)
    >>> len(best)
    4
    >>> all(pd.Series(best).isin(merged.Major_Category.unique()))
    True
    '''
    
    cop = df.copy()
    cop['Employment_rate'] = cop['Employed'] / cop['Total']
    best_emply_rate = cop.groupby('Major_Category')['Employment_rate'].mean().idxmax()
    best_median_sal = cop.groupby('Major_Category')['Median'].median().idxmax()
    best_min_p75 = cop.groupby('Major_Category')['P75th'].min().idxmax()
    year_round = cop.groupby('Major_Category')['Employed_full_time_year_round'].sum().idxmax()
    
    return [best_emply_rate, best_median_sal, best_min_p75, year_round]

# Question 2


def null_and_statistic():
    """
    answers to the two multiple-choice
    questions listed above.

    :Example:
    >>> out = null_and_statistic()
    >>> isinstance(out, list)
    True
    >>> out[0] in [1,2,3]
    True
    >>> out[1] in [1,2]
    True
    """
    
    return [3, 1]


def simulate_null(data):
    """
    simulate_null takes in a dataframe like default, 
    and returns one instance of the test-statistic 
    (difference of means) under the null hypothesis.

    :Example:
    >>> default_fp = os.path.join('data', 'default.csv')
    >>> default = pd.read_csv(default_fp)
    >>> out = simulate_null(default)
    >>> isinstance(out, float)
    True
    >>> 0 <= out <= 1.0
    True
    """
    # shuffle the weights
    shuffled_DEFAULT = (
        data['DEFAULT']
        .sample(replace=False, frac=1)
        .reset_index(drop=True)
    )
    
    # put them in a table
    shuffled = (
        data
        .assign(**{'Shuffled DEFAULT': shuffled_DEFAULT})
    )
    
    # compute the group differences (test statistic!)
    group_means = (
        shuffled
        .groupby('Shuffled DEFAULT')
        .mean()
        .loc[:, 'AVG_PAY_AMT']
    )
    difference = group_means.diff().iloc[-1]

    return difference


def pval_default(data):
    """
    pval_default takes in a dataframe like default, 
    and calculates the p-value for the permutation 
    test using 1000 trials.
    
    :Example:
    >>> default_fp = os.path.join('data', 'default.csv')
    >>> default = pd.read_csv(default_fp)
    >>> out = pval_default(default)
    >>> isinstance(pval, float)
    True
    >>> 0 <= pval <= 0.1
    True
    """
    results = []
    for _ in range(1000):
        results.append(simulate_null(data))
        
    obs = (
        data
        .groupby('DEFAULT')['AVG_PAY_AMT']
        .mean()
        .diff()
        .iloc[-1]
    )
    pval = (pd.Series(results) < obs).mean()
    
    return pval

# Question 3

def identifications():
    """
    Multiple choice response for question X
    >>> out = identifications()
    >>> ans = ['MD', 'MCAR', 'MAR', 'NMAR']
    >>> len(out) == 5
    True
    >>> set(out) <= set(ans)
    True
    """
    #MAR - based on professor
    #MCAR - some students left already, not related to their name/major/color
    #MAR - People with more complicated names more likely to have a nickname; MD is reasonable as well
    #NMAR - if there is no direct translation, the software skips the word
    #MAR - professors that were just hired/haven't taught won't have an avg rec rate
    
    return ['MAR', 'MCAR', 'MD', 'NMAR', 'MAR']


# Question 4

#starter code

def impute_years(cars):
    """
    impute_years takes in a DataFrame of car data
    with missing values and imputes them using the scheme in
    the question.
    :Example:
    >>> fp = os.path.join('data', 'cars.csv')
    >>> df = pd.read_csv(fp)
    >>> out = impute_years(df)
    >>> out['car_year'].dtype == int
    True
    >>> out['car_year'].min() == df['car_year'].min()
    True
    """
    medians = cars.groupby('car_make')['car_year'].median()
    medians = medians.round()
    medians = medians.fillna(medians.median()) #In this dataset, the only McLaren car has no year

    def impute(row):
        if pd.isnull(row['car_year']):
            row['car_year'] = medians[row['car_make']]
        return row
    
    return cars.apply(impute, axis = 1)

# Question 5

def impute_colors(cars):
    """
    impute_colors takes in a DataFrame of car data
    with missing values and imputes them using the scheme in
    the question.
    :Example:
    >>> fp = os.path.join('data', 'cars.csv')
    >>> df = pd.read_csv(fp)
    >>> out = impute_colors(df)
    >>> out.loc[out['car_make'] == 'Toyota'].nunique() == 19
    True
    >>> 'Crimson' in out.loc[out['car_make'] == 'Austin']['car_color'].unique()
    False
    """
    no_color = cars.loc[cars['car_color'].isnull()]
    color_dists = cars.groupby('car_make')['car_color'].value_counts()

    def color_impute(row):
        if pd.isnull(row['car_color']):
            row['car_color'] = color_dists[row['car_make']]\
                .sample(weights = color_dists[row['car_make']].values).index[0]
        return row

    imputed = no_color.apply(color_impute, axis = 1)
    cars[cars['car_color'].isnull()] = imputed
    
    return cars


# Question 7

def match(robots):
    """
    >>> robots1 = "User-Agent: *\\nDisallow: /posts/\\nDisallow: /posts?\\nDisallow: /amzn/click/\\nDisallow: /questions/ask/\\nAllow: /"
    >>> match(robots1)
    False
    >>> robots2 = "User-Agent: *\\nAllow: /"
    >>> match(robots2)
    True
    >>> robots3 = "User-agent: Googlebot-Image\\nDisallow: /*/ivc/*\\nUser-Agent: *\\nAllow: /"
    >>> match(robots3)
    True
    """
    return re.search(r'\bUser-Agent: \*\nAllow: /$',robots) is not None


# Question 8

def extract(text):
    """
    extracts all phone numbers from given 
    text and return the findings as a 
    list of strings
    :Example:
    >>> text1 = "Contact us\\nFinancial Aid and Scholarships Office\\nPhone: (858)534-4480\\nFax: (858)534-5459\\nWebsite: fas.ucsd.edu\\nEmail: finaid@ucsd.edu\\nMailing address:\\n9500 Gilman Drive, Mail Code 0013\\nLa Jolla, CA 92093-0013"
    ['(858)534-4480','(858)534-5459']
    >>> text2 = "Contact us\\nPhone: 858-534-4480\\nFax: 858-534-5459\\nMailing address:\\n9500 Gilman Drive, Mail Code 0013\\nLa Jolla, CA 92093-00130"
    ['858-534-4480','858-534-5459']
    """
    return re.findall('((?:\(\d{3}\)|\d{3})-?\d{3}-?\d{4})',text)


# Question 9


def tfidf_data(sentences):
    """
    tf-idf of the word 'data' in a list of `sentences`.
    """
    words = pd.Series(sentences.str.split().sum())

    tf = sentences.str.count(r'\bdata\b') / (sentences.str.count(' ') + 1)
    idf = np.log(len(sentences) / sentences.str.contains(r'\bdata\b').sum())

    tfidf = tf*idf
    return tfidf

# Question 10

def vectorize(df):
    """
    Create a vector, indexed by the distinct words, with counts of the words in that entry.
    """
    return pd.Series(df['genres'].split('|')).value_counts()

# Question 11


def qualitative_columns():
    return ["CHARSET", "SERVER", "WHOIS_COUNTRY", "WHOIS_STATEPRO"]


# Question 12

def false_consequences():
    """
    
    >>> false_consequences() in range(1, 5)
    True
    """
    
    return 4

def blocked_malicious():
    """
    
    >>> out = blocked_malicious()
    >>> set(out[0]) <= set(range(5))
    True
    >>> 0 <= out[1] <= 1
    True
    """
    
    return ([1, 2, 3], 0.345)

def fairness_claims():
    """
    
    >>> out = fairness_claims()
    >>> set(out[0]) <= set(range(5))
    True
    >>> 0 <= out[1] <= 1
    True
    """
    
    return ([1, 2], 1)
    
# Question 13

def parameters():
    
    params = {
        "clf__n_estimators": [10, 100],
        "clf__max_depth": [10, 50],
        "clf__min_samples_split": [2, 4],
        "clf__class_weight": [{0:0.5}, {0:0.2}, {0:0.1}]
    }
    
    return params

def parameter_search(X, y, pl):
    
    params = parameters()

    recl_grid = GridSearchCV(pl, params, scoring="recall", cv=3)
    prec_grid = GridSearchCV(pl, params, scoring="precision", cv=3)
    f1_grid = GridSearchCV(pl, params, scoring="f1", cv=3)
    
    pl_recl = recl_grid.fit(X, y).best_estimator_
    pl_prec = prec_grid.fit(X, y).best_estimator_
    pl_f1 = f1_grid.fit(X, y).best_estimator_
    
    return [pl_recl, pl_prec, pl_f1]


# Question 14

def age_pairity(X, y, pl, scoring, k):
    
    def age_brackets(ages, k):
        return ages.apply(lambda x: k * (x // (k*365.25) + 1))
    
    results = X.assign(
        malicious=y,
        predicted=pl.predict(X),
        age_bracket=age_brackets(X.WHOIS_AGE_DAYS, k)
    )
    
    return results.groupby("age_bracket").apply(lambda x: scoring(x.malicious, x.predicted))
