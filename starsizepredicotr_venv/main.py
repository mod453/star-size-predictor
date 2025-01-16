from fastapi import FastAPI, File, UploadFile, Query
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

W = 1.982015  # Weight (slope)
b = 9.500380  # Bias (intercept)

@app.get('/')
def default():
    return {'App': 'Running'} # This code defines a basic endpoint in a FastAPI application that responds with a JSON message when accessed. It's a common way to provide a simple health check or status response for your API.

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    df.columns = ['inputs', 'targets']
    df['predictions'] = W * df['inputs'] + b
    output = df.to_csv(index=False).encode('utf-8')
    return StreamingResponse(io.BytesIO(output),
                             media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=predictions.csv"})

@app.post("/plot/")
async def plot(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    plt.figure(figsize=(10, 6))
    plt.scatter(df['inputs'], df['targets'], color='royalblue', label='Actual Targets', marker='x')
    df['predictions'] = W * df['inputs'] + b
    rmse_score = np.mean(np.square(df['predictions'].values - df['targets'].values))
    plt.plot(df['inputs'], df['predictions'], color='k', label='Predictions', linewidth=2)
    plt.title(f'Linear Regression for Stars Data (RMSE: {round(rmse_score, 1)})', color='maroon', fontsize=15)
    plt.xlabel('Brightness', color='m', fontsize=13)
    plt.ylabel('Size', color='m', fontsize=13)
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return StreamingResponse(buf,
                             media_type="image/png",
                             headers={"Content-Disposition": "attachment; filename=plot.png"})

@app.post("/generate_data/")
async def generate_data(num_stars: int = Query(..., gt=0, description="Number of stars to generate")):
    # Generate test data
    X_test = 3 * np.random.rand(num_stars, 1)
    y_test = 9 + 2 * X_test + np.random.rand(num_stars, 1)
    
    
    # Create DataFrame
    df1 = pd.DataFrame(list(zip(X_test.reshape(num_stars,), y_test.reshape(num_stars,))), 
                       columns=['Test Input', 'True Output'])
    
    # Prepare CSV for download
    output = df1.to_csv(index=False).encode('utf-8')
    return StreamingResponse(io.BytesIO(output),
                             media_type="text/csv",
                             headers={"Content-Disposition": f"attachment; filename=test_data_{num_stars}.csv"})