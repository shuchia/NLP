FROM python:3.8-slim

WORKDIR /NLP

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py"]
