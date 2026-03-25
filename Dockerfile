FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip install schedule
COPY . .
CMD ["python", "autorun.py", "--schedule", "--interval", "6", "--capital", "200"]
