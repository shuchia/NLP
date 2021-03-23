MODEL_PATH = "./models/"

MODEL_NAMES = {
    "bart-large-cnn": "facebook/bart-large-cnn",
    "pegasus_xsum ": "google/google_pegasus_xsum",
    "pegasus-multinews": "google/pegasus-multinews",
    "pegasus-reddit_tifu": "google/pegasus-reddit_tifu",
    "pegasus-cnn_dailymail": "google/pegasus-cnn_dailymail",
    "pegasus-large": "google/pegasus-large",
    "pegasus-newsroom": "google/pegasus-newsroom",
    "pegasus-billsum": "google/pegasus-billsum"
}

TOKENIZERS = {
    "facebook/bart-large-cnn": "BartTokenizer"

}

MODELS = {
    "facebook/bart-large-cnn": "BartForConditionalGeneration"
}
