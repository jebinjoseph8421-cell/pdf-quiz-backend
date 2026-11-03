import os
import json

from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_community.vectorstores import FAISS


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Google API key not found in .env file")


def generate_quiz(pdf_file, difficulty, question_count):

    # 1. Extract text from PDF
    reader = PdfReader(pdf_file)

    pdf_text = ""

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pdf_text += text + "\n"

    if not pdf_text.strip():
        raise ValueError("No readable text found in this PDF")

    # 2. Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(pdf_text)

    # 3. Create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    # 4. Create FAISS vector database
    vectorstore = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )

    # 5. Retrieve relevant content
    results = vectorstore.similarity_search(
        "Important concepts and topics in this document",
        k=min(5, len(chunks))
    )

    context = "\n\n".join(
        [result.page_content for result in results]
    )

    # 6. Initialize Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )

    # 7. Generate quiz
    prompt = f"""
    Generate exactly {question_count} multiple-choice questions
    based ONLY on the following document content.

    Difficulty: {difficulty}

    Return valid JSON in this exact format:

    {{
        "quiz": [
            {{
                "question": "Question here",
                "options": {{
                    "A": "Option A",
                    "B": "Option B",
                    "C": "Option C",
                    "D": "Option D"
                }},
                "correct_answer": "A",
                "explanation": "Explanation here"
            }}
        ]
    }}

    Document content:
    {context}
    """

    response = llm.invoke(prompt)

    response_text = response.content

    if isinstance(response_text, list):
        response_text = "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in response_text
        )

    response_text = response_text.strip()

    # Remove Markdown code fences if Gemini adds them
    if response_text.startswith("```"):
        response_text = response_text.split("\n", 1)[1]
        response_text = response_text.rsplit("```", 1)[0]

    quiz_data = json.loads(response_text)

    return quiz_data