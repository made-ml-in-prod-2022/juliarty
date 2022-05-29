import logging
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, PositiveInt
from starlette import status
from starlette.requests import Request
from starlette.responses import JSONResponse

from .config_params import get_config_params
from .model import get_inference_pipeline


logger = logging.getLogger(__name__)
logger.info("Starting app.")
config_path = "config/config.yaml"
config_params = get_config_params(config_path)

app = FastAPI()

try:
    pipeline = get_inference_pipeline(config_params.model_path)
except Exception:
    pipeline = None


class PredictSample(BaseModel):
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=0, le=3)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    exang: int = Field(..., ge=0, le=1)
    slope: int = Field(..., ge=0, le=2)
    ca: int = Field(..., ge=0, le=3)
    thal: int = Field(..., ge=0, le=2)
    oldpeak: float = Field(..., ge=0, le=100)
    age: int = Field(..., ge=0, le=150)
    trestbps: PositiveInt
    thalach: PositiveInt
    chol: PositiveInt


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=jsonable_encoder(
            {"detail": exc, "Error": "Something wrong with data provided"}
        ),
    )


@app.get("/", tags=["home"])
async def home():
    return {"msg": "Hello World"}


@app.post("/predict", tags=["predict"])
async def predict(sample: PredictSample):
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The model is not ready.",
        )

    logger.info(sample)
    result = pipeline.predict(pd.DataFrame({k: [v] for k, v in sample.dict().items()}))
    return {"result": result.tolist()[0]}


@app.get("/health", tags=["status"], status_code=status.HTTP_200_OK)
async def health():
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The model is not ready.",
        )

    return "That's fine"
