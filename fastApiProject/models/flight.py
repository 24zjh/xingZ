from pydantic import BaseModel

class Flight(BaseModel):
    flight_number: str
    status: str
    departure: str
    arrival: str
    departure_time: str
    arrival_time: str
