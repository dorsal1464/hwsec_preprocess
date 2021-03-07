# from utils import traces, Y, SAMPLES
from scipy.io import loadmat
from SNR import SNR
import numpy as np
import matplotlib.pyplot as plt
import hdf5storage
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request

executor = ThreadPoolExecutor(8, "test")

URLS = ['http://www.foxnews.com/',
        'http://www.cnn.com/',
        'http://europe.wsj.com/',
        'http://www.bbc.co.uk/',
        'http://some-made-up-domain.com/',
        'http://www.google.com']

# Retrieve a single page and report the URL and contents


def load_url(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as conn:
        return conn.read()


ans1 = executor.map(load_url, URLS)
print(zip(ans1))
# We can use a with statement to ensure threads are cleaned up promptly
datamap = list()

for url in URLS:
    datamap.append(executor.submit(load_url, url, 30))

for future in datamap:
    print(future.result())



