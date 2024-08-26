import requests
from models.weather import Weather

def get_weather(city: str) -> Weather:
    # Mock response, replace with actual API call
    response = {
        "city": city,
        "temperature": "30°C",
        "humidity": "60%",
        "wind": "北风",
        "description": "晴天"
    }
    return Weather(**response)
