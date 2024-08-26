from pydantic import BaseModel

class TextRequest(BaseModel):
    content: str

class TextResponse(BaseModel):
    result: str
