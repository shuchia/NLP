from transformers import BartTokenizer, BartForConditionalGeneration
import nltk
import torch
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')


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


def generate_summary(tokenizer, model, nested_sentences):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    summaries = []
    for nested in nested_sentences:
        input_tokenized = tokenizer.encode(' '.join(nested), truncation=True, return_tensors='pt')
        input_tokenized = input_tokenized.to(device)
        summary_ids = model.to(device).generate(input_tokenized,
                                                length_penalty=3.0)
        output = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                  summary_ids]
        summaries.append(output)
    summaries = [sentence for sublist in summaries for sentence in sublist]
    return summaries


def inference(model_name, article_text):
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    nested = nest_sentences(article_text)

    summarized_text = generate_summary(tokenizer, model, nested)
    nested_summ = nest_sentences(' '.join(summarized_text))
    return generate_summary(tokenizer, model, nested_summ)
    #return nested
