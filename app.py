import sys
import os
import certifi
import pandas as pd
import pymongo

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import Response, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.constants.traning_pipeline import (
    DATA_INGESTION_COLLECTION_NAME,
    DATA_INGESTION_DATABASE_NAME,
)

# -------------------------------------------------
# ENV & DB
# -------------------------------------------------
load_dotenv()
mongo_db_url = os.getenv("MONGODB_URL_KEY")
print(mongo_db_url)

ca = certifi.where()
client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)

print("🚀 Network Security ML App is starting...")
print(f"📂 Connected to MongoDB database: {DATA_INGESTION_DATABASE_NAME}")
print(f"📂 Using collection: {DATA_INGESTION_COLLECTION_NAME}")
print("✅ FastAPI app initialized")


database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

# -------------------------------------------------
# FASTAPI APP (MUST BE BEFORE ROUTES)
# -------------------------------------------------
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates
templates = Jinja2Templates(directory="./templates")

# -------------------------------------------------
# ROUTES
# -------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head>
            <title>Network Security ML App</title>
        </head>
        <body style="text-align:center; margin-top:50px;">
            <h1>Network Security ML Project 🚀</h1>
            <p><a href="/docs">Open API Docs</a></p>
            <p><a href="/train">Train Model</a></p>
        </body>
    </html>
    """

@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training is successful")
    except Exception as e:
        raise NetworkSecurityException(e, sys)

@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        print(f"📄 Uploaded file: {file.filename}")
        print(f"✅ Loaded {len(df)} rows of data for prediction")
        print("🔍 Sample data:")
        print(df.head())

         # Drop the target or any unseen columns (like 'Result')
        if "Result" in df.columns:
            df = df.drop("Result", axis=1)
            print("⚠️ Dropped column 'Result' for prediction")

        preprocessor = load_object("final_model/preprocessor.pkl")
        model = load_object("final_model/model.pkl")

        network_model = NetworkModel(
            preprocessor=preprocessor,
            model=model
        )

        y_pred = network_model.predict(df)
        df["predicted_column"] = y_pred
        example_output = df.head(10)
        example_output.to_csv("prediction_output/example_output.csv", index=False)
        print("✅ Saved example predictions to 'prediction_output/example_output.csv'")

        import matplotlib.pyplot as plt

        def save_table_image(df):
            fig, ax = plt.subplots(figsize=(12, 2))  # adjust size as needed
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=df.head(10).values,
                             colLabels=df.head(10).columns,
                             loc='center')
            plt.savefig("images/prediction_table.png", bbox_inches='tight')
            plt.close()
            print("📸 Saved prediction table image to 'images/prediction_table.png'")

        save_table_image(df)
        df.to_csv("prediction_output/output.csv", index=False)

        table_html = df.to_html(classes="table table-striped")
        return templates.TemplateResponse(
            "table.html",
            {"request": request, "table": table_html}
        )

    except Exception as e:
        raise NetworkSecurityException(e, sys)

# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8000)
