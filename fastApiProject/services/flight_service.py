from typing import Optional
from models.flight import Flight

def get_flight_status(flight_number: str, start: Optional[str], end: Optional[str]) -> Flight:
    # Mock response, replace with actual API call
    response = {
        "flight_number": flight_number,
        "status": "准时",
        "departure": "北京首都国际机场",
        "arrival": "上海浦东国际机场",
        "departure_time": "10:00",
        "arrival_time": "12:00"
    }
    return Flight(**response)
