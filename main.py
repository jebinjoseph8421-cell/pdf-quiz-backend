from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO

from rag import generate_quiz

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
   allow_origins=[
    "http://localhost:5173",
    "https://pdf-quiz-frontend.vercel.app"
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "PDF Quiz Generator Backend is running!"
    }


@app.post("/generate-quiz")
async def generate_quiz_api(
    file: UploadFile = File(...),
    difficulty: str = Form(...),
    question_count: int = Form(...)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file."
        )

    try:
        pdf_content = await file.read()

        quiz_data = generate_quiz(
            BytesIO(pdf_content),
            difficulty,
            question_count
        )

        return quiz_data

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )