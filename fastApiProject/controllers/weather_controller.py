from fastapi import APIRouter, Query
from services.weather_service import get_weather
from models.weather import Weather

router = APIRouter()

@router.get("/", response_model=Weather)
async def read_weather(city: str = Query(..., description="The name of the city to query weather for")):
    return get_weather(city)
