import math

from numpy import ones, array, sqrt
from scipy.sparse import csr_matrix

def intersect(a, b):
    return len(a.intersection(b))

def inter(a, b):
    return a.intersection(b)

def jaccard(a, b):
    intersection = float(len(a.intersection(b)))
    return intersection / (len(a) + len(b) - intersection)

def dice(a, b):
    intersection = float(len(a.intersection(b)))
    return 2 * intersection / (len(a) + len(b))

def ochiai(a, b):
    intersection = float(len(a.intersection(b)))
    return intersection / sqrt(len(a) * len(b))

# cosine distances
def cosine(a, b):
    return csr_matrix.dot(a, b.T)[0, 0] / (norm2(a) * norm2(b))

def norm2(artist):
    return sqrt((artist.data ** 2).sum())

def smoothed_cosine(a, b):
    smoothing = 20.0
    overlap = csr_matrix.dot(binarize(a), binarize(b).T)[0, 0]

    return (overlap / (smoothing + overlap)) * cosine(a, b)

def binarize(artist):
    ret = csr_matrix(artist)
    ret.data = ones(len(artist.data))
    return ret

def tfidf(a, b, idf):
    return cosine(tfidf_weight(a, idf), tfidf_weight(b, idf))

def tfidf_weight(artist, idf):
    ret = csr_matrix(artist)
    ret.data = array([sqrt(plays) * idf[userid]
                      for plays, userid in zip(artist.data, artist.indices)])
    return ret
