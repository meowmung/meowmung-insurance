FROM python:3.9

COPY requirements.txt requirements.txt 

RUN pip install -r requirements.txt

WORKDIR /meowmung-insurance

COPY . /meowmung-insurance

EXPOSE 80

CMD ["uvicorn", "FastAPI.advancedApp:app", "--host", "0.0.0.0", "--port", "80"]