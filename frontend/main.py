import time

import requests
import streamlit as st
import pandas as pd

TYPES = {
    "general": "bart-large-cnn",
    "composition 6": "composition_vii",
    "feathers": "feathers",
    "la_muse": "la_muse",
    "mosaic": "mosaic",
    "starry night": "starry_night",
    "the scream": "the_scream",
    "the wave": "the_wave",
    "udnie": "udnie",
}

st.set_option("deprecation.showfileUploaderEncoding", False)

st.title("Text Summarization")

file = st.file_uploader("Upload an excel file", type="xlsx")

contentType = st.selectbox("Choose the type", [i for i in TYPES.values()])

if st.button("Summarize"):
    if file is not None and contentType is not None:
        files = {"file": file.getvalue()}
        df = pd.read_excel(file.read(), index_col=None, header=None)
        print(len(df))
        # displayedUrl = sheet.cell_value(1,0)
        res = requests.post(f"http://backend:8080/{contentType}", files=files)
        file_path = res.json()
        print(file_path)
        sentences = []
        fileHandle = open(file_path.get('name'), 'r')

        url = file_path.get("url")
        st.write(url)
        st.write(fileHandle.read())

        displayed = 1
        df1 = df.iloc[2:]
        total = len(df1)
        displayed_urls = [url]

        st.write("Generating other summaries...")

        while displayed <= total:
            for i in range(len(df1)):
                url = df1.iat[i, 0]
                print(url)
                path = f"{file_path.get('name').split('.')[0]}_{i}.txt"
                print(path)
                if url not in displayed_urls:
                    try:
                        fileHandle = open(path, 'r')
                        st.write(url)
                        st.write(fileHandle.read())
                        displayed += 1
                        displayed_urls.append(url)
                    except:
                        pass
