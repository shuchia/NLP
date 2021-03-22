import asyncio
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# Webscrapping using BeautifulSoup, not yet implemented
import bs4 as bs  # beautifulsource4
import urllib.request
import re
import pandas as pd

import uvicorn
from fastapi import File
from fastapi import FastAPI
from fastapi import UploadFile
from fastapi.logger import logger
# ... other imports
import logging.config
import numpy as np

import config
from inference import NLP
import os

# Natural Language Processing Libraries
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("Starting server")

app = FastAPI()
nlp = NLP()


@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}


def get_text(url):
    headers = {
        'User-Agent': 'Mozilla/5.0'}
    opener = urllib.request.URLopener()
    opener.addheader('User-Agent', 'Mozilla/5.0')

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    scraped_data = urllib.request.urlopen(req, timeout=20)

    article = scraped_data.read()

    parsed_article = bs.BeautifulSoup(article, 'lxml')
    parsed_article = bs.BeautifulSoup(article, 'lxml')

    paragraphs = parsed_article.find_all('p')

    article_text = ""

    for p in paragraphs:
        article_text += ' ' + p.text
    formatted_article_text = re.sub(r'\n|\r', ' ', article_text)
    formatted_article_text = re.sub(r' +', ' ', formatted_article_text)
    formatted_article_text = formatted_article_text.strip()
    return formatted_article_text


@app.post("/{contentType}")
async def get_summary(contentType: str, file: UploadFile = File(...)):
    logger.info("file " + file.filename)
    df = pd.read_excel(file.file.read(), index_col=None, header=None)
    logger.info(len(df))
    model_name = config.MODEL_NAMES[contentType]
    url = df.iat[1, 0]
    logger.info("url " + url)
    article_text = get_text(url)
    start = time.time()
    summary = nlp.inference(model_name, article_text)
    name = f"/storage/{str(uuid.uuid4())}.txt"
    logger.info(f"name: {name}")
    # file1 = open(name, "w+")
    with open(name, 'w+') as file1:
        for listItem in summary:
            file1.write('%s\n' % listItem)
    df1 = df.iloc[2:]
    asyncio.create_task(generate_remaining_summaries(model_name, name, df1))
    return {"url": url, "name": name, "time": time.time() - start}


async def generate_remaining_summaries(model_name, name, sheet):
    logger.info("model_name in async " + model_name)
    executor = ProcessPoolExecutor()
    event_loop = asyncio.get_event_loop()
    await event_loop.run_in_executor(
        executor, partial(generate_summary, model_name, name, sheet)
    )


def generate_summary(model_name, name, df):
    logger.info("model_name in generate_summary " + model_name)
    for ind in range(len(df)):
        url = df.iat[ind, 0]
        logger.info("url " + url)
        article_text = get_text(url)
        logger.info(article_text)
        summary = nlp.inference(model_name, article_text)
        name = name.split(".")[0]
        name = f"{name.split('_')[0]}_{ind}.txt"
        logger.info("name " + name)
        with open(name, 'w+') as file1:
            for listItem in summary:
                file1.write('%s\n' % listItem)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
