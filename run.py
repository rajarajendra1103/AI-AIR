import uvicorn
import webbrowser
import os
import sys

if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8000
    print(f"Starting AI Air Quality & Weather Forecasting Platform at http://{host}:{port}")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
