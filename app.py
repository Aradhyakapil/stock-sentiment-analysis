from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pickle
import os
from pathlib import Path
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize the FastAPI app
app = FastAPI()

# Mount static files for CSS and other assets
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define paths to model and vectorizer files
model_path = Path(r"C:\Users\hp\Desktop\stock\models\best_model.pkl")
vectorizer_path = Path(r"C:\Users\hp\Desktop\stock\models\tfidf_vectorizer.pkl")

# Load the machine learning model and vectorizer
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

with open(vectorizer_path, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Initialize NLTK's SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Define a Pydantic model for incoming data (for the prediction form)
class PredictionRequest(BaseModel):
    headline: str

# Initialize Jinja2Templates 
templates = Jinja2Templates(directory="templates") 

# Serve the main HTML page
@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):  # Pass the request object
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

# Define a route to handle form submissions and predictions
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, headline: str = Form(...)):
    # Vectorize the input text
    transformed_headline = vectorizer.transform([headline])
    
    # Add sentiment features
    sentiment_features = np.array(list(sia.polarity_scores(headline).values())).reshape(1, -1)  # Convert to list first
    #sentiment_features = np.array(sia.polarity_scores(headline).values()).reshape(1, -1)
    
    # Combine TF-IDF and sentiment features
    combined_features = np.hstack((transformed_headline.toarray(), sentiment_features))
    
    # Predict sentiment (e.g., 0 = Down/Same, 1 = Up)
    prediction = model.predict(combined_features)
    result = "The stock price will go up!" if prediction[0] == 1 else "The stock price will remain the same or will go down."

    # Render the result in the HTML template
    return templates.TemplateResponse("index.html", {"request": request, "result": result})  # Pass request object


if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000)
