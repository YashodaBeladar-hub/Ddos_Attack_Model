import logging
import uuid
import warnings
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from datetime import datetime, timezone

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.base")

# Set up logging
logging.basicConfig(
    filename="request_logs.log",  # Log to a file
    filemode="a",  # Append mode
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    level=logging.INFO,  # Set logging level
)

logger = logging.getLogger(__name__)

app = FastAPI()

# Load the standard scaler
scaler = joblib.load("standard_scaler.joblib")

# Load models and encoders
models = {
    "dns": (
        joblib.load("Models/DDoS_DNS.joblib"),
        joblib.load("Encoders/encoder(dns).joblib"),
    ),
    "ldap": (
        joblib.load("Models/DDoS_LDAP.joblib"),
        joblib.load("Encoders/encoder(ldap).joblib"),
    ),
    "mssql": (
        joblib.load("Models/DDoS_MSSQL.joblib"),
        joblib.load("Encoders/encoder(mssql).joblib"),
    ),
    "netbios": (
        joblib.load("Models/DDoS_NetBIOS.joblib"),
        joblib.load("Encoders/encoder(netbios).joblib"),
    ),
    "ntp": (
        joblib.load("Models/DDoS_NTP.joblib"),
        joblib.load("Encoders/encoder(ntp).joblib"),
    ),
    "portmap": (
        joblib.load("Models/DDoS_Portmap.joblib"),
        joblib.load("Encoders/encoder(portmap).joblib"),
    ),
    "snmp": (
        joblib.load("Models/DDoS_SNMP.joblib"),
        joblib.load("Encoders/encoder(snmp).joblib"),
    ),
    "ssdp": (
        joblib.load("Models/DDoS_SSDP.joblib"),
        joblib.load("Encoders/encoder(ssdp).joblib"),
    ),
    "syn": (
        joblib.load("Models/DDoS_SYN.joblib"),
        joblib.load("Encoders/encoder(syn).joblib"),
    ),
    "tftp": (
        joblib.load("Models/DDoS_TFTP.joblib"),
        joblib.load("Encoders/encoder(tftp).joblib"),
    ),
    "udp": (
        joblib.load("Models/DDoS_UDP.joblib"),
        joblib.load("Encoders/encoder(udp).joblib"),
    ),
    "udplag": (
        joblib.load("Models/DDoS_UDPLag.joblib"),
        joblib.load("Encoders/encoder(udplag).joblib"),
    ),
}


class PredictionRequest(BaseModel):
    features: list


def log_request(
    request_id,
    timestamp,
    protocol,
    input_features,
    prediction=None,
    probability=None,
    error=None,
):
    """Log the request details using the logging library."""
    logger.info(
        {
            "request_id": request_id,
            "timestamp": timestamp,
            "protocol": protocol,
            "input_features": input_features,
            "prediction": prediction,
            "probability": probability,
            "error": error,
        }
    )


@app.post("/predict/")
async def predict(request: PredictionRequest):
    # Generate a unique request ID
    request_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    if len(request.features) != 69:
        log_request(
            request_id,
            timestamp,
            None,
            request.features,
            error="Bad input: Expected 69 features",
        )
        raise HTTPException(status_code=400, detail="Bad input: Expected 69 features")

    max_probability = -1
    max_prediction = None
    max_protocol = None

    try:
        features = np.array(request.features).reshape(1, -1)
        scaled_features = scaler.transform(features)

        # Iterate over the models and make predictions
        for protocol, (model, encoder) in models.items():
            prediction = model.predict(scaled_features)
            probability = model.predict_proba(scaled_features).max()

            if probability > max_probability:
                max_prediction = prediction[0]
                max_probability = probability
                max_protocol = protocol

        # If no valid prediction found, set max_protocol to 'BENIGN'
        if max_prediction == 0:
            max_protocol = "BENIGN"
            max_probability = 1.0  # Assuming max certainty for benign

        if max_protocol is None:
            log_request(
                request_id,
                timestamp,
                None,
                request.features,
                error="No valid prediction",
            )
            raise HTTPException(status_code=500, detail="No valid prediction")

        # Log the successful prediction
        log_request(
            request_id=request_id,
            timestamp=timestamp,
            protocol=max_protocol,
            input_features=request.features,
            prediction=int(max_prediction),
            probability=float(max_probability),
        )

        return {
            "protocol": max_protocol,
            "prediction": int(max_prediction),
            "probability": float(max_probability),
        }
    except Exception as e:
        log_request(request_id, timestamp, None, request.features, error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Error during prediction: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
