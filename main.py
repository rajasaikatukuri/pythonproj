from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

# Initialize the FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use a specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
try:
    model = joblib.load("logistic_regression_pipeline.joblib")
    print("Logistic Regression Pipeline model loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")
    raise

# Define the input schema
class InputFeatures(BaseModel):
    area_mean: float
    area_se: float
    area_worst: float
    compactness_mean: float
    compactness_se: float
    compactness_worst: float
    concave_points_mean: float
    concave_points_se: float
    concave_points_worst: float
    concavity_mean: float
    concavity_se: float
    concavity_worst: float
    fractal_dimension_mean: float
    fractal_dimension_se: float
    fractal_dimension_worst: float
    perimeter_mean: float
    perimeter_se: float
    perimeter_worst: float
    radius_mean: float
    radius_se: float
    radius_worst: float
    smoothness_mean: float
    smoothness_se: float
    smoothness_worst: float
    symmetry_mean: float
    symmetry_se: float
    symmetry_worst: float
    texture_mean: float
    texture_se: float
    texture_worst: float

@app.get("/")
def read_root():
    return {"message": "Logistic Regression Prediction API is running!"}

@app.post("/predict")
def predict(input_data: InputFeatures):
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data.dict()])

        # Perform prediction
        prediction = model.predict(input_df)
        diagnosis = "Malignant" if prediction[0] == 1 else "Benign"

        return {"diagnosis": diagnosis}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
