import json
import os

from keras.models import load_model
import pickle
from sklearn.cluster import KMeans


def feature_engineer(dataset):
    X = dataset[dataset.columns[2:]]
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

# make prediction
def predict(dataset):
    model_repo = os.environ['MODEL_REPO']
    if model_repo:
        file_path = os.path.join(model_repo, "model.pkl")
        model = pickle.load(open(file_path, 'rb'))
        val_set2 = dataset.copy()
        features = dataset[dataset.columns[2:]]
        result = model.predict(features)
        # y_classes = result.argmax(axis=-1)
        val_set2['RUL'] = result.tolist()
        dic = val_set2.to_dict(orient='records')
        return json.dumps(dic, indent=4, sort_keys=False)
    else:
        return json.dumps({'message': 'A model cannot be found.'},
                          sort_keys=False, indent=4)
