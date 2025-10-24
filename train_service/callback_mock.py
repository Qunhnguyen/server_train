from fastapi import FastAPI, Request
import uvicorn, json

app = FastAPI()

@app.post("/api/train/callback")
async def cb(req: Request):
    data = await req.json()
    print("\n=== CALLBACK RECEIVED ===")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return {"ok": True}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8099)
