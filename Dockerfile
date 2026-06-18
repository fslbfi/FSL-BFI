FROM python:3.10-slim

# tshark for live packet capture (non-interactive: don't prompt about dumpcap setuid)
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tshark && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install only the realtime app runtime deps (fastapi, uvicorn, websockets, torch, numpy, scikit-learn)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "-m", "realtime_app.main"]
