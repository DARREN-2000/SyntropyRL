FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install uv && uv pip install --system -e .
CMD ["python", "-m", "syntropyrl"]
