import os
import sys
import json
import re
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
    allow_origins=["*"],
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
    Uses Gemini structured output to identify exact error line numbers.
    Falls back safely to extracting from traceback if AI fails.
    """

    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        prompt = f"""
Analyze the Python CODE and TRACEBACK.
Return ONLY valid JSON in this format:

{{ "error_lines": [line_numbers] }}

Important:
- Extract line numbers from the original user code.
- Ignore any internal framework files.

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

        parsed = json.loads(response.text)
        return parsed.get("error_lines", [])

    except Exception:
        # ✅ SAFE FALLBACK:
        # Extract only from user code reference: File "<string>", line X
        match = re.search(r'File "<string>", line (\d+)', traceback_text)
        if match:
            return [int(match.group(1))]
        return []

# ---------------------------------------------------
# Health Check Endpoint (Required for Render)
# ---------------------------------------------------

@app.get("/")
def health_check():
    return {"status": "ok"}

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