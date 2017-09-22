from collections import defaultdict
from numpy import zeros, log, array
from scipy.sparse import csr_matrix

import os
import pandas
import sys

class Dataset(object):
    def __init__(self, dataset):

        self.data = dataset

        self.item_sets = dict((item, set(users)) for item, users in \
        self.data.groupby('item')['user'])

        self.user_items = dict((user, set(items)) for user, items in \
        self.data.groupby('user')['item'])

        self.user_sets = self.data.set_index('user').apply(tuple, 1) \
             .groupby(level=0).agg(lambda x: list(x.values)) \
             .to_dict()

        # map user -> id
        userids = defaultdict(lambda: len(userids))
        itemids = defaultdict(lambda: len(itemids))

        self.data['userid'] = self.data['user'].map(userids.__getitem__)
        self.data['itemid'] = self.data['item'].map(itemids.__getitem__)

        # create sparse vector for each item, user
        self.items = dict((item,
                             csr_matrix((array(group['rating']),
                                         (zeros(len(group)),
                                         group['userid'])),
                                        shape=[1, len(userids)]))
                            for item, group in self.data.groupby('item'))

        self.users = dict((user,
                             csr_matrix((array(group['rating']),
                                         (zeros(len(group)),
                                         group['itemid'])),
                                        shape=[1, len(itemids)]))
                            for user, group in self.data.groupby('user'))

        N = len(self.items)
        self.idf = [1. + log(N / (1. + p)) for p in self.data.groupby('userid').size()]
        self.average_plays = self.data['rating'].sum() / float(N)

def read_millionsong_data(dataset, metadataset):
    # read in triples of user/artist/playcount from the millionsong dataset
    data = pandas.read_table(dataset, names=['user', 'song_id', 'count'])
    # read metadata
    metadata = pandas.read_csv(metadataset)

    data = pandas.merge(data, metadata.drop_duplicates(['song_id']),
    on='song_id', how='left')

    data['song'] = data['title'].map(str) + " - " + data['artist_name']
    data.drop(['title', 'artist_name'], inplace=True, axis=1)

    # clean up
    data.drop(['song_id', 'release', 'year'], inplace=True, axis=1)
    # data.drop(['artist'], inplace=True, axis=1)

    data = data.rename(index=str, columns={'song': 'item'})

    data['rating'] = [10 if x > 10 else x for x in data['count']]
    data.drop(['count'], inplace=True, axis=1)

    return data.dropna(axis=1, how='all')
