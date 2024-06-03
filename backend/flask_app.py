# flask_app.py
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import json
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from flask import Flask, request, jsonify
from flask_cors import CORS

import emoji
from bs4 import BeautifulSoup
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

contraction_mapping = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "this's": "this is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "here's": "here is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
    "u.s": "america",
    "e.g": "for example",
}
punct = [
    ",",
    ".",
    '"',
    ":",
    ")",
    "(",
    "-",
    "!",
    "?",
    "|",
    ";",
    "'",
    "$",
    "&",
    "/",
    "[",
    "]",
    ">",
    "%",
    "=",
    "#",
    "*",
    "+",
    "\\",
    "•",
    "~",
    "@",
    "£",
    "·",
    "_",
    "{",
    "}",
    "©",
    "^",
    "®",
    "`",
    "<",
    "→",
    "°",
    "€",
    "™",
    "›",
    "♥",
    "←",
    "×",
    "§",
    "″",
    "′",
    "Â",
    "█",
    "½",
    "à",
    "…",
    "“",
    "★",
    "”",
    "–",
    "●",
    "â",
    "►",
    "−",
    "¢",
    "²",
    "¬",
    "░",
    "¶",
    "↑",
    "±",
    "¿",
    "▾",
    "═",
    "¦",
    "║",
    "―",
    "¥",
    "▓",
    "—",
    "‹",
    "─",
    "▒",
    "：",
    "¼",
    "⊕",
    "▼",
    "▪",
    "†",
    "■",
    "’",
    "▀",
    "¨",
    "▄",
    "♫",
    "☆",
    "é",
    "¯",
    "♦",
    "¤",
    "▲",
    "è",
    "¸",
    "¾",
    "Ã",
    "⋅",
    "‘",
    "∞",
    "∙",
    "）",
    "↓",
    "、",
    "│",
    "（",
    "»",
    "，",
    "♪",
    "╩",
    "╚",
    "³",
    "・",
    "╦",
    "╣",
    "╔",
    "╗",
    "▬",
    "❤",
    "ï",
    "Ø",
    "¹",
    "≤",
    "‡",
    "√",
]
punct_mapping = {
    "‘": "'",
    "₹": "e",
    "´": "'",
    "°": "",
    "€": "e",
    "™": "tm",
    "√": " sqrt ",
    "×": "x",
    "²": "2",
    "—": "-",
    "–": "-",
    "’": "'",
    "_": "-",
    "`": "'",
    "“": '"',
    "”": '"',
    "“": '"',
    "£": "e",
    "∞": "infinity",
    "θ": "theta",
    "÷": "/",
    "α": "alpha",
    "•": ".",
    "à": "a",
    "−": "-",
    "β": "beta",
    "∅": "",
    "³": "3",
    "π": "pi",
    "!": " ",
}
mispell_dict = {
    "colour": "color",
    "centre": "center",
    "favourite": "favorite",
    "travelling": "traveling",
    "counselling": "counseling",
    "theatre": "theater",
    "cancelled": "canceled",
    "labour": "labor",
    "organisation": "organization",
    "wwii": "world war 2",
    "citicise": "criticize",
    "youtu ": "youtube ",
    "Qoura": "Quora",
    "sallary": "salary",
    "Whta": "What",
    "narcisist": "narcissist",
    "howdo": "how do",
    "whatare": "what are",
    "howcan": "how can",
    "howmuch": "how much",
    "howmany": "how many",
    "whydo": "why do",
    "doI": "do I",
    "theBest": "the best",
    "howdoes": "how does",
    "mastrubation": "masturbation",
    "mastrubate": "masturbate",
    "mastrubating": "masturbating",
    "pennis": "penis",
    "Etherium": "Ethereum",
    "narcissit": "narcissist",
    "bigdata": "big data",
    "2k17": "2017",
    "2k18": "2018",
    "qouta": "quota",
    "exboyfriend": "ex boyfriend",
    "airhostess": "air hostess",
    "whst": "what",
    "watsapp": "whatsapp",
    "demonitisation": "demonetization",
    "demonitization": "demonetization",
    "demonetisation": "demonetization",
}


def clean_text(text):
    """Clean emoji, Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers."""
    text = emoji.demojize(text)
    text = re.sub(r"\:(.*?)\:", "", text)
    text = str(text).lower()  # Making Text Lowercase
    text = re.sub("\[.*?\]", "", text)
    # The next 2 lines remove html text
    text = BeautifulSoup(text, "lxml").get_text()
    text = re.sub("https?://\S+|www\.\S+", "", text)
    text = re.sub("<.*?>+", "", text)
    text = re.sub("\n", "", text)
    text = re.sub("\w*\d\w*", "", text)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",", "'")
    text = re.sub(r"[^a-zA-Z?.!,¿']+", " ", text)
    return text


def clean_contractions(text, mapping):
    """Clean contraction using contraction mapping"""
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    for word in mapping.keys():
        if "" + word + "" in text:
            text = text.replace("" + word + "", "" + mapping[word] + "")
    # Remove Punctuations
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    text = re.sub(r"([?.!,¿])", r" \1 ", text)
    text = re.sub(r'[" "]+', " ", text)
    return text


def clean_special_chars(text, punct, mapping):
    """Cleans special characters present(if any)"""
    for p in mapping:
        text = text.replace(p, mapping[p])

    for p in punct:
        text = text.replace(p, f" {p} ")

    specials = {"\u200b": " ", "…": " ... ", "\ufeff": "", "करना": "", "है": ""}
    for s in specials:
        text = text.replace(s, specials[s])

    return text


def correct_spelling(x, dic):
    """Corrects common spelling errors"""
    for word in dic.keys():
        x = x.replace(word, dic[word])
    return x


def remove_space(text):
    """Removes awkward spaces"""
    # Removes awkward spaces
    text = text.strip()
    text = text.split()
    return " ".join(text)


def text_preprocessing_pipeline(text):
    """Cleaning and parsing the text."""
    text = clean_text(text)
    text = clean_contractions(text, contraction_mapping)
    text = clean_special_chars(text, punct, punct_mapping)
    text = correct_spelling(text, mispell_dict)
    text = remove_space(text)
    return text


app = Flask(__name__)
CORS(app)

# Load the model and tokenizer
model_path = "bert_model.h5"
print(f"Loading model from {model_path}")
model = tf.keras.models.load_model(
    model_path, custom_objects={"TFBertModel": TFBertModel}
)
print("Model loaded successfully")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
print("Tokenizer loaded successfully")

instagram_account_id = "17841416265505184"
access_token = "EAAOArR24FGUBOZCA5dY8cnNNfyrVWgsZBAUG975u8yGdyh6gYignpZCJZCCcGd9wDXl3rPiOIGoZCAWUhPTKIONGsYopaMa1LcGgMT51KlnCYlbAE4wbZCiZCZA1CtQK4kg6louXmYQxaZBCDWvXFA50xjZC23x4tjhgB3cLUeG2RcT5N5rXV4NfAm2vZCPFXHXvZC97LOYt7cAYZBBrCZC4wyfpOIOSeAV7AZD"


def get_user_info_and_posts(username, instagram_account_id, access_token):
    ig_params = {
        "fields": "business_discovery.username("
        + username
        + "){media.limit(100){caption}}",
        "access_token": access_token,
    }
    endpoint = f"https://graph.facebook.com/v19.0/{instagram_account_id}"
    response = requests.get(endpoint, params=ig_params)
    return response.json()


def fetch_instagram_captions(username, instagram_account_id, access_token):
    print(f"Fetching Instagram captions for user: {username}")
    data = get_user_info_and_posts(username, instagram_account_id, access_token)

    if "business_discovery" not in data:
        print("Failed to fetch captions. Response:", data)
        return []

    business_discovery = data["business_discovery"]
    captions = [
        media["caption"]
        for media in business_discovery["media"]["data"]
        if "caption" in media
    ]
    print(f"Fetched {len(captions)} captions")
    concatenated_caption = " ".join(captions)

    return concatenated_caption


def prepare_data(input_text, tokenizer):
    input_text = text_preprocessing_pipeline(input_text)
    token = tokenizer.encode_plus(
        input_text,
        max_length=256,
        truncation=True,
        padding="max_length",
        add_special_tokens=True,
        return_tensors="tf",
    )
    return {
        "input_ids": tf.cast(token.input_ids, tf.int32),
        "attention_mask": tf.cast(token.attention_mask, tf.int32),
    }


def filter_invalid_tokens(token_ids, max_token_id):
    valid_token_ids = tf.minimum(token_ids, max_token_id)
    return valid_token_ids


def predict_interests(caption):
    print("Preprocessing and tokenizing captions")
    max_length = 256

    tokenized_data = prepare_data(caption, tokenizer)
    input_ids = tokenized_data["input_ids"]
    attention_mask = tokenized_data["attention_mask"]

    input_ids = filter_invalid_tokens(input_ids, 28995)

    print("Generating predictions")
    outputs = model.predict([input_ids, attention_mask])
    print("Predictions generated successfully")

    return outputs


def analyze_predictions(predictions, class_names):
    thresholded_predictions = np.where(predictions > 0.5, 1, 0)
    significant_classes = {
        class_names[i]
        for i in range(len(class_names))
        if np.any(thresholded_predictions[:, i] == 1)
    }

    return significant_classes


def convert_to_native_type(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_native_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_type(i) for i in obj]
    else:
        return obj


@app.route("/predict", methods=["POST"])
def predict():
    username = request.json.get("username")
    if not username:
        return jsonify({"error": "No username provided"}), 400

    concatenated_caption = fetch_instagram_captions(
        username, instagram_account_id, access_token
    )
    if not concatenated_caption:
        return jsonify({"error": "Failed to fetch captions or no captions found"}), 400

    classes = [
        "Business and Industry",
        "Entertainment",
        "Outdoors",
        "Technology",
        "family and relationships",
        "fitness and wellness",
        "food and drink",
        "hobbies and activities",
        "shopping and fashion",
        "sports",
    ]

    predictions = predict_interests(concatenated_caption)
    significant_classes = analyze_predictions(predictions, classes)

    predictions_native = convert_to_native_type(predictions)

    return jsonify(
        {
            "classes": list(significant_classes),
            "predictions": {
                class_name: predictions_native[0][i]
                for i, class_name in enumerate(classes)
            },
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
