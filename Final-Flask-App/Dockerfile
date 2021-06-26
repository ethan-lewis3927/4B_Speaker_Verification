FROM python:3.6
RUN apt-get update && apt-get upgrade -y
RUN pip install --upgrade pip
COPY . /app
WORKDIR /app
RUN pip3 --no-cache-dir install -r requirements.txt
RUN pip3 install librosa
RUN apt-get --yes install libsndfile1
EXPOSE 5000
CMD ["python", "application.py"]