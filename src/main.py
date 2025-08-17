# Import the API endpoints from the dedicated module
from api_endpoints import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
