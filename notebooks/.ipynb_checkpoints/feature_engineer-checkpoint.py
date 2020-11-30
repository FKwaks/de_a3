import json
import os
import pickle
import copy 


def clean_data(dataset):
    X = dataset[dataset.columns[2:]].copy()
    engine = dataset.iloc[:,0].to_list()
    cycle = dataset.iloc[:,1].to_list()

    features = dataset.columns[2:]
    for feature in features:
        # Creating min, max and delta variables
        X['max_' + feature] = dataset.groupby('engine_id')[feature].cummax()
        X['min_' + feature] = dataset.groupby('engine_id')[feature].cummin()

        X['delta_' + feature] = dataset.groupby('engine_id')[feature].diff()
        X['delta_' + feature].fillna(0, inplace=True)

    X.insert(0, 'engine_id', engine)
    X.insert(1, 'cycle', cycle)
    return X
