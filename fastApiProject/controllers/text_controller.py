from fastapi import APIRouter, HTTPException
from models.text import TextRequest, TextResponse
from services.text_service import get_text_response

router = APIRouter()

@router.post("/", response_model=TextResponse)
def process_text(request: TextRequest):
    try:
        response = get_text_response(request.content)
        return TextResponse(result=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
