from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
import pickle
import pandas as pd
import uvicorn

app = FastAPI()

# Load the pipeline model
with open('pipeline model.pkl', 'rb') as f:
    pipeline = pickle.load(f)

templates = Jinja2Templates(directory='templates')

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@app.post('/predict')
def predict(
    request: Request,
    Gender: str = Form(...),
    Age: float = Form(...),
    Height: float = Form(...),
    Weight: float = Form(...),
    Duration: float = Form(...),
    Heart_Rate: float = Form(default=0.0),  # Default value added
    Body_Temp: float = Form(...)
):
    # Corrected from pd.DateFrame to pd.DataFrame
    sample = pd.DataFrame({
        'Gender': [Gender],
        'Age': [Age], 
        'Height': [Height], 
        'Weight': [Weight], 
        'Duration': [Duration],
        'Heart_Rate': [Heart_Rate], 
        'Body_Temp': [Body_Temp]
    }, index=[0]) 

    result = pipeline.predict(sample)[0]

    return templates.TemplateResponse('result.html', {'request': request, 'calories': result})

if __name__ == '__main__':  # Corrected the condition
    uvicorn.run(app, port=8000)