import os
import sys
import json
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
# ✅ CORS ENABLED (Required for testing)
# ---------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OK for testing
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
    """
    Executes Python code and returns exact stdout or exact traceback.
    """
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
    Always fails safely (never crashes API).
    """

    try:
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

        prompt = f"""
You are analyzing Python execution errors.

Given the CODE and its TRACEBACK below,
extract ONLY the line number(s) from the original code
where the error occurred.

Important:
- Return ONLY valid JSON.
- No explanations.
- No markdown.
- No extra text.

Format:
{{ "error_lines": [line_numbers] }}

CODE:
{code}

TRACEBACK:
{traceback_text}
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
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

        # Safe JSON parsing (prevents 500 errors)
        data = json.loads(response.text)
        return data.get("error_lines", [])

    except Exception as e:
        # Never crash API
        print("AI error analysis failed:", str(e))
        return []


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