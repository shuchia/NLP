from transformers import BartTokenizer, BartForConditionalGeneration
import nltk
import torch
import logging.config
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def nest_sentences(document):
    nested = []
    sent = []
    length = 0
    for sentence in nltk.sent_tokenize(document):
        length += len(sentence)
        if length < 1024:
            sent.append(sentence)
        else:
            nested.append(sent)
            sent = [sentence]
            length = len(sentence)

    if sent:
        nested.append(sent)

    return nested


class NLP:
    def __init__(self):
        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

    def generate_summary(self, model, nested_sentences):
        logger.info("Inside inference generate summary")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        summaries = []
        for nested in nested_sentences:
            input_tokenized = self.encode(' '.join(nested), truncation=True, return_tensors='pt')
            input_tokenized = input_tokenized.to(device)
            summary_ids = model.to(device).generate(input_tokenized,
                                                    length_penalty=3.0)
            output = [self.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                      summary_ids]
            summaries.append(output)
        summaries = [sentence for sublist in summaries for sentence in sublist]
        return summaries

    def inference(self, model_name, article_text):
        logger.info("Inside inference after model load")
        nested = nest_sentences(article_text)

        summarized_text = self.generate_summary(self.tokenizer, self.model, nested)
        nested_summ = nest_sentences(' '.join(summarized_text))
        return self.generate_summary(self.tokenizer, self.model, nested_summ)
    # return nested

# def inference(model_name, article_text):
#     # tokenizer = BartTokenizer.from_pretrained(model_name)
#     # model = BartForConditionalGeneration.from_pretrained(model_name)
#     nested = nest_sentences(article_text)
#
#     # summarized_text = generate_summary(tokenizer, model, nested)
#     # nested_summ = nest_sentences(' '.join(summarized_text))
#     # return generate_summary(tokenizer, model, nested_summ)
#     return nested
