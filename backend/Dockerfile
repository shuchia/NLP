FROM tiangolo/uvicorn-gunicorn:python3.8
COPY ./app /app
WORKDIR /NLP

ENV MODEL_DIR=/home/ubuntu/NLP/backend/models/

RUN apt-get update
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "app/main.py"]
