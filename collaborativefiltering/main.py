import sys
import os
import pandas
import numpy
import pickle
from dataset import data
from dataset import *
from collections import defaultdict, OrderedDict
from operator import *
from util.sims import *

PATH = os.getcwd() + '/collaborativefiltering/dataset/'
TRAINING_DATA = "./bot/training_data"


def similar_user_items(data, user, k=10):
    """ User-User Recommendation
    Finds similar items from similar users to a given users items. """

    user_items = [x for (x, y) in data.user_sets[user]]
    user_set = set(user_items)
    print('User %s') % user
    print_items(user_items[:k])

    similar_users = most_similar_users(data, user_set, k)
    top_user, top_rating, top_sum = k_closest_items(data, similar_users)

    return top_user, top_rating, top_sum

def print_items(items):
    """ Print a list of similar items (recommendations) for a user. """
    print("\nItems:")
    print('*' * 24)
    for item in items:
        print item
    print('\n')

# Find most similar users
def most_similar_users(data, user_set, k):
    """ User-User Recommendation.
    Finds similar users for this user's set of items. """
    sims = defaultdict()
    for user, items in data.user_items.iteritems():
        num_items = len(items)
        other_items = items - user_set
        left = num_items - len(other_items)
        if left > 0 and other_items is not []:
            sims[user] = (left, other_items)

    k_closest = sorted(sims, key=lambda x: x[0], reverse=True)

    ret = list()
    for user in k_closest:
        ret.append(user)
    return ret

def get_random_user(data):
    """ Return a random userid in the available dataset. """
    users = data.user_sets.keys()
    randuser = numpy.random.randint(0, len(users), 1)[0]
    return users[randuser]

def sims(item_sets, user_set, users, items, idf):
    """ Return lambda functions for calculating several similarity metrics. """
    return {
        'intersect': lambda x: intersect(item_sets[x], user_set),
        'ochiai': lambda x: ochiai(item_sets[x], user_set),
        'jaccard': lambda x: jaccard(item_sets[x], user_set),
        'cosine': lambda x: cosine(items[x], users),
        'smoothed_cosine': lambda x: smoothed_cosine(items[x], users),
        'tfidf': lambda x: tfidf(items[x], users, idf),
    }

def k_closest_items(data, users):
    """ Calculate k closest items for a given set of users. """
    ret = defaultdict()
    for u in users:
        for item in data.user_sets[u]:
            title = item[0]
            rating = item[1]
            if ret.get(title) is None:
                ret[title] = (rating, 1)
            else:
                ret[title] = (ret.get(title)[0] + rating, ret.get(title)[1] + 1)

    ret = sorted(ret.iteritems(), key=lambda (k, v): v[0], reverse=True)
    ret = knns(ret, 10)
    return ret


def find_similar_items_to(data, item, min_users=5):
    """ Item-Item recommendation. Find similar items to this item. """
    metrics = generate_item(data, item)
    for metric, similar in metrics.iteritems():
        print '\n'
        print '*' * 24
        print 'Metric: ' + metric
        print 'Recs: \n'
        for x in similar:
            print x['rank'], x['item'], x['score']

def similar_items(data, item):
    """ Item-Item recommendation. Return similar items to this item. """
    ret = defaultdict()
    for u in data.item_sets[item]:
        for item in data.user_sets[u]:
            title = item[0]
            rating = item[1]
            if ret.get(title) is None:
                ret[title] = (rating, 1)
            else:
                ret[title] = (ret.get(title)[0] + rating, ret.get(title)[1] + 1)

    ret = sorted(ret.iteritems(), key=lambda (k, v): v[0], reverse=True)
    top_k_users, _, _ = knns(ret, 10)
    return top_k_users

def generate_item(data, item):
    """ Print recommendations with different similarity metric data for an item. """
    print 'Item-Item recommendations for %s' % item

    users = data.items[item].tocsc()
    user_set = data.item_sets[item]
    user_count = len(data.user_sets)

    similarities = sims(data.item_sets, user_set, users, data.items, data.idf)

    metrics, best = {}, set()
    inters = lambda x: inter(data.item_sets[x], user_set)
    others = similar_items(data, item)
    similar = [(inters(other), other) for other in others]
    similar.sort(reverse=True)

    others = similar_items(data, item)
    for name, sim in similarities.iteritems():
        similar = [(sim(other), other) for other in others]
        similar.sort(reverse=True)
        metrics[name] = similar
        best.update(n for _, n in similar[:10])

    output = {}
    for metric, similar in metrics.iteritems():
        filtered = []
        for i, (score, name) in enumerate(similar):
            if name in best:
                filtered.append({'item': title(name),
                                 'score': score,
                                 'rank': i + 1})

        output[metric] = filtered
    return output

def knns(candidates, k):
    """ Returns the knns for a set of candidates.
    Strategies: Most popular among users, most listened count, highest scores.
    """
    try:
        knns = [(title, ratings[0], ratings[1], ratings[0] / ratings[1]) \
        for title, ratings in candidates]
    except TypeError:
        try:
            knns = [(title, ratings[0], ratings[1], ratings[0] / ratings[1]) \
            for title, ratings in candidates.iteritems() if not numpy.isnan(title)]
        except TypeError:
            pass

    popular_users_count = sorted(knns, key=itemgetter(2, 1), reverse=True)
    count_popular_users = sorted(knns, key=itemgetter(1, 2), reverse=True)
    best_score = sorted(knns, key=itemgetter(3), reverse=True)
    top_k_users = [title for title, point, people, avg in popular_users_count][:k]
    top_k_count = [title for title, point, people, avg in count_popular_users][:k]
    top_k_score = [title for title, point, people, avg in best_score][:k]
    return top_k_users, top_k_count, top_k_score

def create_dataset():
    """ Creates a dataset object from given millionsong subset data and metadata """
    dataset = PATH + '10000.txt'
    metadata = PATH + 'song_data.csv'
    # create dataframe
    df = data.read_millionsong_data(dataset, metadata)
    # create dataset object
    dataset = data.Dataset(df)
    print("loaded dataset")
    return dataset

def title(s):
    """ Returns a formatted title given a string """
    return s.decode("utf8").title()

def get_songs_by_artist(data, s):
    """ Returns correctly formatted song titles for a given artist """
    items = [k for k, v in data.item_sets.iteritems() if s.lower() in k.lower()]
    return items


def get_song(data, s):
    """ Returns a correctly formatted title for a queried song """
    items = [k for k, v in data.item_sets.iteritems() if s.lower() in k.lower()]
    if len(items) > 0:
        return items[0]
    else:
        return None

def main():
    dataset = create_dataset()

    # get a random user
    user = get_random_user(dataset)
    # user-user recommendation
    top_user, top_rating, top_sum = similar_user_items(dataset, user)

    print("Top similar user recommendations:")
    print_items(top_user)

    print("Top rated item recommendations:")
    print_items(top_rating)

    print("Top scored item recommendations:")
    print_items(top_sum)
    # get a song

    song = get_song(dataset, 'The Scientist - Coldplay')

    # get all songs by artist
    all_songs = get_songs_by_artist(dataset, 'lady gaga')
    print all_songs

    find_similar_items_to(dataset, song)
    find_similar_items_to(dataset, all_songs[1])


if __name__ == '__main__':
    main()
