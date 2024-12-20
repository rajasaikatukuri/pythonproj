{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "69e3f6c9-9abf-4357-ad4a-57b7d628a953",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting fastapi\n",
      "  Downloading fastapi-0.115.6-py3-none-any.whl.metadata (27 kB)\n",
      "Collecting starlette<0.42.0,>=0.40.0 (from fastapi)\n",
      "  Downloading starlette-0.41.3-py3-none-any.whl.metadata (6.0 kB)\n",
      "Requirement already satisfied: pydantic!=1.8,!=1.8.1,!=2.0.0,!=2.0.1,!=2.1.0,<3.0.0,>=1.7.4 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from fastapi) (2.10.4)\n",
      "Requirement already satisfied: typing-extensions>=4.8.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from fastapi) (4.12.2)\n",
      "Requirement already satisfied: annotated-types>=0.6.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from pydantic!=1.8,!=1.8.1,!=2.0.0,!=2.0.1,!=2.1.0,<3.0.0,>=1.7.4->fastapi) (0.7.0)\n",
      "Requirement already satisfied: pydantic-core==2.27.2 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from pydantic!=1.8,!=1.8.1,!=2.0.0,!=2.0.1,!=2.1.0,<3.0.0,>=1.7.4->fastapi) (2.27.2)\n",
      "Requirement already satisfied: anyio<5,>=3.4.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from starlette<0.42.0,>=0.40.0->fastapi) (4.7.0)\n",
      "Requirement already satisfied: idna>=2.8 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from anyio<5,>=3.4.0->starlette<0.42.0,>=0.40.0->fastapi) (3.10)\n",
      "Requirement already satisfied: sniffio>=1.1 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from anyio<5,>=3.4.0->starlette<0.42.0,>=0.40.0->fastapi) (1.3.1)\n",
      "Downloading fastapi-0.115.6-py3-none-any.whl (94 kB)\n",
      "Downloading starlette-0.41.3-py3-none-any.whl (73 kB)\n",
      "Installing collected packages: starlette, fastapi\n",
      "Successfully installed fastapi-0.115.6 starlette-0.41.3\n"
     ]
    }
   ],
   "source": [
    "!pip install fastapi"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "6d0c450d-d49d-43d6-bee7-cfb4d4efba1b",
   "metadata": {},
   "outputs": [],
   "source": [
    "from fastapi import FastAPI\n",
    "from pydantic import BaseModel\n",
    "import joblib\n",
    "import numpy as np\n",
    "\n",
    "# Load the saved model\n",
    "model = joblib.load(\"/Users/rajasaikatukuri/Downloads/pythonproject/final_model.pkl\")\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "539f85ad-476a-4a15-b5c9-1021ade300f6",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "# Initialize the FastAPI app\n",
    "app = FastAPI()\n",
    "\n",
    "# Define the input schema using Pydantic with all dataset features\n",
    "class InputFeatures(BaseModel):\n",
    "    mean_radius: float\n",
    "    mean_texture: float\n",
    "    mean_perimeter: float\n",
    "    mean_area: float\n",
    "    mean_smoothness: float\n",
    "    mean_compactness: float\n",
    "    mean_concavity: float\n",
    "    mean_concave_points: float\n",
    "    mean_symmetry: float\n",
    "    mean_fractal_dimension: float\n",
    "    radius_error: float\n",
    "    texture_error: float\n",
    "    perimeter_error: float\n",
    "    area_error: float\n",
    "    smoothness_error: float\n",
    "    compactness_error: float\n",
    "    concavity_error: float\n",
    "    concave_points_error: float\n",
    "    symmetry_error: float\n",
    "    fractal_dimension_error: float\n",
    "    worst_radius: float\n",
    "    worst_texture: float\n",
    "    worst_perimeter: float\n",
    "    worst_area: float\n",
    "    worst_smoothness: float\n",
    "    worst_compactness: float\n",
    "    worst_concavity: float\n",
    "    worst_concave_points: float\n",
    "    worst_symmetry: float\n",
    "    worst_fractal_dimension: float\n",
    "\n",
    "# Root endpoint for health check\n",
    "@app.get(\"/\")\n",
    "def read_root():\n",
    "    return {\"message\": \"Breast Cancer Prediction API is running!\"}\n",
    "\n",
    "# Prediction endpoint\n",
    "@app.post(\"/predict\")\n",
    "def predict(input_data: InputFeatures):\n",
    "    # Convert input features into a DataFrame for transformations\n",
    "    input_df = pd.DataFrame([input_data.dict()])\n",
    "\n",
    "    # Add meaningful ratios\n",
    "    input_df['area_to_radius_mean'] = input_df['mean_area'] / input_df['mean_radius']\n",
    "\n",
    "    # Polynomial features\n",
    "    input_df['radius_mean_squared'] = input_df['mean_radius'] ** 2\n",
    "\n",
    "    # Log transformations\n",
    "    input_df['log_area_mean'] = np.log1p(input_df['mean_area'])\n",
    "\n",
    "    # Convert the transformed data into a NumPy array\n",
    "    input_array = input_df.values\n",
    "\n",
    "    # Perform prediction\n",
    "    prediction = model.predict(input_array)\n",
    "\n",
    "    return {\"prediction\": int(prediction[0])}\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "2f0dee4e-7092-4ad8-872b-d8649d9633f7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: uvicorn in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (0.34.0)\n",
      "Requirement already satisfied: click>=7.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from uvicorn) (8.1.7)\n",
      "Requirement already satisfied: h11>=0.8 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from uvicorn) (0.14.0)\n"
     ]
    }
   ],
   "source": [
    "!pip install uvicorn\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "f03bd4ed-e947-4628-9a45-b7428b3ec3c8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[32mINFO\u001b[0m:     Will watch for changes in these directories: ['/Users/rajasaikatukuri/Downloads/pythonproject']\n",
      "\u001b[32mINFO\u001b[0m:     Uvicorn running on \u001b[1mhttp://127.0.0.1:8000\u001b[0m (Press CTRL+C to quit)\n",
      "\u001b[32mINFO\u001b[0m:     Started reloader process [\u001b[36m\u001b[1m14465\u001b[0m] using \u001b[36m\u001b[1mStatReload\u001b[0m\n",
      "\u001b[31mERROR\u001b[0m:    Error loading ASGI app. Could not import module \"untitled1\".\n",
      "^C\n",
      "\u001b[32mINFO\u001b[0m:     Stopping reloader process [\u001b[36m\u001b[1m14465\u001b[0m]\n"
     ]
    }
   ],
   "source": [
    "!uvicorn untitled1:app --reload --port 8000\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cdbbc221-510e-43f1-ab9f-f42b56b27091",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
