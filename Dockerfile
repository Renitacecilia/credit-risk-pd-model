FROM python:3.11-slim

# libgomp1 is the OpenMP runtime XGBoost needs on Linux
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY credit_risk.py .
COPY data/ data/
# Train inside the image so the saved model always matches the installed library versions
RUN python credit_risk.py train

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "credit_risk:app", "--host", "0.0.0.0", "--port", "8000"]
