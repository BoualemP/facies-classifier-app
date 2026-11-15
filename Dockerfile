FROM python:3.12-slim

RUN useradd --create-home --uid 1000 appuser
WORKDIR /home/appuser/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser app ./app
COPY --chown=appuser static ./static

USER appuser
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
