from flask import Flask, request
from json import loads
from predictor.anime_predictor import get_predictor

app = Flask(__name__)

model = get_predictor('./predictor/final_model')


@app.route('/', methods=['POST'])
def service():
    r = dict(loads(request.get_json()))
    # questions are in correct order
    # TODO: use something more reliable than a dict in that regard
    return model.predict(list(r.values()))
