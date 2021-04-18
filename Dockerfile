FROM tensorflow/tensorflow 


ENV APP_HOME /app
WORKDIR $APP_HOME

RUN pip install -U pip

RUN apt-get update -qq && apt-get install -y -q libsndfile1

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

RUN pip install librosa

COPY . ./

# Run
CMD streamlit run app.py --server.enableCORS false --server.port $PORT

