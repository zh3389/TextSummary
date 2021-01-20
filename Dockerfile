FROM python:3.6.12-slim
MAINTAINER zhanghao <zhanghao_3389@163.com>

RUN mkdir /TextSummary
WORKDIR /TextSummary
ADD . /TextSummary

RUN python -m pip install --upgrade pip \
    pip install -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]