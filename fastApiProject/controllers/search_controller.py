from fastapi import APIRouter, Query
from typing import List
from services.search_service import search_baidu
from models.search_result import SearchResult

router = APIRouter()

@router.get("/", response_model=List[SearchResult])
async def read_search_results(
    query: str = Query(..., description="The search query"),
    top_k: int = Query(2, description="The number of search results to return")
):
    return search_baidu(query, top_k)
