import os
import sys
import traceback
from io import StringIO
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()

app = FastAPI()

# ---------------------------------------------------
# ✅ CORS ENABLED (Required)
# ---------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# Request Model
# ---------------------------------------------------

class CodeRequest(BaseModel):
    code: str


# ---------------------------------------------------
# Structured Output Model (AI Response)
# ---------------------------------------------------

class ErrorAnalysis(BaseModel):
    error_lines: List[int]


# ---------------------------------------------------
# Tool Function: Execute Python Code
# ---------------------------------------------------

def execute_python_code(code: str) -> dict:
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        exec(code)
        output = sys.stdout.getvalue()
        return {"success": True, "output": output}

    except Exception:
        output = traceback.format_exc()
        return {"success": False, "output": output}

    finally:
        sys.stdout = old_stdout


# ---------------------------------------------------
# AI Error Analysis (ONLY when needed)
# ---------------------------------------------------

def analyze_error_with_ai(code: str, traceback_text: str) -> List[int]:
    """
    Uses Gemini with structured output to identify exact error line numbers.
    """

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    prompt = f"""
Analyze this Python code and its error traceback.
Identify the exact line number(s) where the error occurred.

CODE:
{code}

TRACEBACK:
{traceback_text}

Return only the JSON object with the error line numbers.
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "error_lines": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.INTEGER),
                    )
                },
                required=["error_lines"],
            ),
        ),
    )

    # Validate structured output using Pydantic
    result = ErrorAnalysis.model_validate_json(response.text)

    return result.error_lines


# ---------------------------------------------------
# FastAPI Endpoint
# ---------------------------------------------------

@app.post("/code-interpreter")
def run_code(request: CodeRequest):

    # 1️⃣ Execute Tool
    execution = execute_python_code(request.code)

    # 2️⃣ If success → return exact output
    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"]
        }

    # 3️⃣ If error → invoke AI
    error_lines = analyze_error_with_ai(
        request.code,
        execution["output"]
    )

    # 4️⃣ Return exact traceback unchanged
    return {
        "error": error_lines,
        "result": execution["output"]
    }