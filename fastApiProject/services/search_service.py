from typing import List
from models.search_result import SearchResult

def search_baidu(query: str, top_k: int) -> List[SearchResult]:
    # Mock response, replace with actual API call
    response = [
        {"title": "北京天气预报", "link": "https://www.baidu.com/link?url=...", "description": "未来一周北京的天气情况..."},
        {"title": "上海天气预报", "link": "https://www.baidu.com/link?url=...", "description": "未来一周上海的天气情况..."}
    ]
    return [SearchResult(**res) for res in response[:top_k]]
