FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SEPSIS_ENVIRONMENT=production \
    SEPSIS_DEMO_PUBLIC_MODE=true

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

COPY streamlit_app.py ./
COPY src ./src
COPY .streamlit ./.streamlit
COPY public_artifacts ./public_artifacts
COPY data/demo ./data/demo

EXPOSE 8501

CMD ["sh", "-c", "streamlit run streamlit_app.py --server.headless true --server.address 0.0.0.0 --server.port ${PORT:-8501}"]
