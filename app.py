import json
import re
from typing import List

import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.security import check_password_hash
from flask import Flask, request, jsonify, make_response
from tqdm import tqdm
nlp_model_ru = spacy.load("ru_core_news_sm")


app = Flask(__name__)
app.json.sort_keys = False
app.config['SECRET_KEY'] = 'djI&^%fg2uh/ko4psij'

# параметры запуска ApI
HOST = '0.0.0.0'
PORT = 5002
API_KEY_HASH = 'scrypt:32768:8:1$0MjrGlmyFk3F4eGK$74ecf2b72fc0f48425935c7e49e05b331f121e7dd84b8d27e49d1a04c408a6c1b00233be6a8d04752e33b5cdce763fc5c85db5468af912b6dd916691c6cc3d9e'
# сколько наиболее похожих текстов выводить
TOP_N = 3


def process_text(text: str) -> List[str]:
    """Функция очистки текста"""
    # очистка от лишних символов (цифры, знаки препинания, символы иных диапазонов utf-8)
    text = re.sub('[^a-zа-яё ]', '', text.lower()).strip()
    # приведение к нижнему регистру
    doc = nlp_model_ru(text)
    # удаление стоп слов (token.is_stop) и лемматизация
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    return tokens


def parse_text_from_request(request_data):
    """Обработка базовых случаев ошибок"""
    # проверка соблюдения формата json
    try:
        data = json.loads(request_data)
    except json.JSONDecodeError:
        return jsonify({"Error": f"Передан некорректный json"}), 400
    text = data.get("text", "")
    # проверка наличия необходимого поля (текст) в запросе
    if not text:
        return jsonify({"Error": "Не передан текст"}), 400
    # проверка наличия api-ключа
    if 'api_key' not in data:
        return jsonify({"Error": "Не передан api-ключ"}), 400
    # проверка совпадения api-ключа
    if not check_password_hash(API_KEY_HASH, data['api_key']):
        return jsonify({"Error": "Передан недействительный api-ключ"}), 400
    return text


@app.route("/process", methods=["POST"])
def process_route():
    """Обработка текстовых запросов"""
    # Обработка базовых случаев ошибок
    parse_result = parse_text_from_request(request.data)
    if not isinstance(parse_result, str):
        return parse_result

    return jsonify({"result": process_text(parse_result)})


@app.route("/search", methods=["POST"])
def search_route():
    """Поиск ТОП-N наиболее схожих текстов для поискового запроса"""
    # Обработка базовых случаев ошибок
    parse_result = parse_text_from_request(request.data)
    if not isinstance(parse_result, str):
        return parse_result

    # сбор нормализованных слова из запроса
    query_tokens = process_text(parse_result)
    # сравнение запроса с описаниями
    query_vector = vectorizer.transform([query_tokens])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    # индексы топ-n наиболее схожих текстов
    top_indices = np.argpartition(-similarities, TOP_N)[:TOP_N]

    results = [{"text": descriptions[i], "score": similarities[i]} for i in top_indices]
    return jsonify({"result": results})


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'Error': 'Not found'}), 404)


if __name__ == "__main__":
    # spacy модель для русского языка (будет использована для лемматизации)
    nlp_model_ru = spacy.load("ru_core_news_sm")

    # отзывы на товары (по которым будет производиться поиск)
    with open("data.json", "r", encoding="utf-8") as file:
        descriptions = json.load(file)

    # lowercase=False, т.к. текст уже конвертирован в нижний регистр
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x, lowercase=False)
    # по примерам текстов собираем нормализованные слова
    documents_tokens = [process_text(description) for description in tqdm(descriptions)]

    # строим tf-idf индекс
    tfidf_matrix = vectorizer.fit_transform(documents_tokens)

    app.run(host='0.0.0.0', port=PORT, debug=True, use_reloader=False)
