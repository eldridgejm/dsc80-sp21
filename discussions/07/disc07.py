import requests


def url_list():
    """
    A list of urls to scrape.

    :Example:
    >>> isinstance(url_list(), list)
    True
    >>> len(url_list()) > 1
    True
    """

    return ...


def request_until_successful(url, N):
    """
    impute (i.e. fill-in) the missing values of each column 
    using the last digit of the value of column A.

    :Example:
    >>> resp = request_until_successful('http://quotes.toscrape.com', N=1)
    >>> resp.ok
    True
    >>> resp = request_until_successful('http://example.webscraping.com/', N=1)
    >>> isinstance(resp, requests.models.Response) or (resp is None)
    True
    """

    return ...
