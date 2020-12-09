from catboost import CatBoostClassifier


def anime_predict(book, english, twoch, games, vegan, path="final_model"):
    classifier = get_predictor(path)
    return classifier.predict([book, english, twoch, games, vegan])


def get_predictor(path='final_model'):
    classifier = CatBoostClassifier()
    classifier.load_model(path)

    return classifier
