# syntax=docker/dockerfile:1
FROM python:3.10-alpine
WORKDIR /web_app
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
RUN apk add build-base gfortran cmake openblas-dev blas-dev pkgconfig
RUN apk add --no-cache gcc musl-dev linux-headers
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 3000
COPY web_app .
<<<<<<< HEAD
<<<<<<< HEAD
CMD ["python", "./app/run.py"]
=======
CMD ["python", "./app/run.py"]
>>>>>>> 3ced3fdf8f216556bd3bf3af9adcc824bcd49cba
=======
CMD ["python", "./app/run.py"]
>>>>>>> 3ced3fdf8f216556bd3bf3af9adcc824bcd49cba
