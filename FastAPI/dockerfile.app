FROM python:3.9

COPY FastAPI/requirements.txt requirements.txt

RUN pip install -r requirements.txt

WORKDIR /advanced

COPY . /advanced

EXPOSE 80

CMD ["uvicorn", "FastAPI.recommendApp:app", "--host", "0.0.0.0", "--port", "80"]

