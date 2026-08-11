import json

import boto3


ENDPOINT_NAME = "country-risk-prediction-v7-v3"
REGION = "us-east-1"


PAYLOAD = {
    "dataframe_records": [
        {
            "gdp_per_capita": 15000.0,
            "inflation": 3.0,
            "life_expectancy": 75.0,
            "population": 50000000.0,
            "population_growth": 1.0,
            "unemployment": 6.0,
            "exports": 25.0,
            "gdp_lag1": 14500.0,
            "inflation_lag1": 3.2,
            "life_expectancy_lag1": 74.8,
            "economic_risk": 0.30,
            "governance_risk": 0.25,
        }
    ]
}


def invoke_endpoint(
    endpoint_name=ENDPOINT_NAME,
    region=REGION,
    payload=PAYLOAD,
):
    runtime = boto3.client(
        "sagemaker-runtime",
        region_name=region,
    )

    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload),
    )

    return json.loads(
        response["Body"].read().decode("utf-8")
    )


def main():
    result = invoke_endpoint()

    print("Endpoint:", ENDPOINT_NAME)
    print("Prediction:", result["predictions"])


if __name__ == "__main__":
    main()