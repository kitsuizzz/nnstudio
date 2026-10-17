FROM python:3.13-slim

WORKDIR /app

COPY app ./app
COPY pyproject.toml README.md ./

# Install the project (pulls runtime deps); the app package ends up in site-packages.
RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
