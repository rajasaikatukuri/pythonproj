# Use the official Python slim image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy application files
COPY main.py .
COPY logistic_regression_pipeline.joblib .

# Install dependencies
# Install dependencies
RUN pip install fastapi uvicorn pandas numpy joblib pydantic scikit-learn mlflow


# Expose the port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
