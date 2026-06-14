FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ghost_listener.py ghost_db.py ghost_utils.py ./

VOLUME /app/data

CMD ["python", "ghost_listener.py", "--interval", "300", "-o", "/app/data/ghost_alerts.json"]
