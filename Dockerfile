FROM python:3.12

WORKDIR /app

COPY requirements.txt .

ENV PYTHONPATH="/app"

RUN pip --no-cache-dir install -r requirements.txt

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]