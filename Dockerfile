FROM python:3.6.8
RUN mkdir APP
WORKDIR APP
COPY requirements.txt .
COPY *.py ./
RUN pip install -r requirements.txt
ENTRYPOINT [ "python", "produce_images.py" ]