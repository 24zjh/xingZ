from fastapi import APIRouter, Query
from typing import Optional
from services.flight_service import get_flight_status
from models.flight import Flight

router = APIRouter()

@router.get("/", response_model=Flight)
async def read_flight_status(
    flight_number: str = Query(..., description="The flight number to query"),
    start: Optional[str] = Query(None, description="Flight start date"),
    end: Optional[str] = Query(None, description="Flight end date")
):
    return get_flight_status(flight_number, start, end)
