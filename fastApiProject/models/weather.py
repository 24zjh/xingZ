from pydantic import BaseModel

class Weather(BaseModel):
    city: str
    temperature: str
    humidity: str
    wind: str
    description: str
