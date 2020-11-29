import pandas as pd
from flask import Flask, json, request, Response, stream_with_context
import _pickle as cPickle
from google.cloud import storage
from urllib.parse import urlparse
from io import BytesIO


from notebooks import feature_engineer

app = Flask(__name__)
app.config["DEBUG"] = True

#test
@app.route('/', methods=['GET'])
def test():
    return {'hello':'world'}

@app.route('/predict/', methods=['POST'])
def predict_perf():
    content = request.get_json()
    df = pd.read_json(json.dumps(content), orient='records')
    df_features = feature_engineer.clean_data(df)
    
    model_store_path = 'gs://de_a3v2/model_store/vanilla/vanilla_gbr.pickle'
    
    parse = urlparse(url=model_store_path, allow_fragments = False)
    if parse.path[0] =='/':
        model_path = parse.path[1:]
    client = storage.Client()
    bucket = client.get_bucket(parse.netloc)
    blob = bucket.get_blob(model_path)
    if blob is None:
        raise AttributeError('No files to download') 
    model_bytestream = BytesIO(blob.download_as_string())
    model = cPickle.load(model_bytestream)
    
    x_predict = df_features[df_features.columns[2:]]
    js = model.predict(x_predict)
    resp = Response(stream_with_context(js), status=200, mimetype='application/json')
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'POST'
    resp.headers['Access-Control-Max-Age'] = '1000'
    return resp

app.run(host='0.0.0.0', port=5000)
