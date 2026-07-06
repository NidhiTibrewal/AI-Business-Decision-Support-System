# Decision-Centric Inventory Allocation Platform
# Runs the FastAPI service by default. Override CMD to run the Streamlit
# dashboard instead (see the commented line at the bottom).

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Generate the synthetic dataset at build time so the container is runnable
# standalone. Swap this for a real data-loading step in production.
RUN python data/generate_synthetic_data.py

EXPOSE 8000
EXPOSE 8501

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]

# To run the dashboard instead:
#   docker run -p 8501:8501 <image> streamlit run app/dashboard.py --server.address 0.0.0.0
