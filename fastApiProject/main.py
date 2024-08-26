from fastapi import FastAPI
from controllers import weather_controller, flight_controller, search_controller,text_controller

app = FastAPI()

app.include_router(weather_controller.router, prefix="/api/weather", tags=["weather"])
app.include_router(flight_controller.router, prefix="/api/flights", tags=["flights"])
app.include_router(search_controller.router, prefix="/api/baidu_search", tags=["search"])
app.include_router(text_controller.router, prefix="/api/text", tags=["text"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", port=8000, reload=True)
# /docs api文档