import json
import sys

import mlflow
import pandas as pd
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response


MODEL_PATH = "/opt/ml/model"

app = FastAPI()

print("=" * 70)
print("Country Risk Prediction - SageMaker Inference Server")
print("=" * 70)
print(f"Loading MLflow model from: {MODEL_PATH}")

model = mlflow.pyfunc.load_model(MODEL_PATH)

print("Model loaded successfully.")
print("=" * 70)


@app.get("/ping")
def ping():
    """
    SageMaker health check.

    Return HTTP 200 only when the model has been loaded successfully.
    """
    if model is None:
        return Response(status_code=503)

    return Response(status_code=200)


@app.post("/invocations")
async def invocations(request: Request):
    """
    SageMaker inference endpoint.
    """

    payload = await request.json()

    if "dataframe_records" in payload:

        dataframe = pd.DataFrame(
            payload["dataframe_records"]
        )

    elif "dataframe_split" in payload:

        split_data = payload["dataframe_split"]

        dataframe = pd.DataFrame(
            data=split_data["data"],
            columns=split_data["columns"],
            index=split_data.get("index"),
        )

    else:

        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    "Request must contain either "
                    "'dataframe_records' or "
                    "'dataframe_split'."
                )
            },
        )

    try:

        predictions = model.predict(dataframe)

        return JSONResponse(
            content={
                "predictions": predictions.tolist()
            }
        )

    except Exception as exc:
        import traceback

        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        )




def main():

    if len(sys.argv) < 2 or sys.argv[1] != "serve":

        print(
            "Usage: python serve.py serve"
        )

        sys.exit(1)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
    )


if __name__ == "__main__":
    main()
