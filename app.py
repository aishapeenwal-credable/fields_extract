import os
import tempfile
import certifi
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import requests
from pdfminer.high_level import extract_text as pdfminer_extract_text

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, origins=[
    "https://preview-31310e4f--agreement-navigator-portal.lovable.app",
    "https://preview--agreement-navigator-portal.lovable.app",
    "http://localhost:3000"
])

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_MODEL = os.getenv("TOGETHER_LLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")

from parameter_config import parameter_categories

def extract_text(file_path, max_chars=8000):
    ext = file_path.lower().split('.')[-1]

    if ext == "pdf":
        try:
            text = pdfminer_extract_text(file_path)
            if not text or len(text.strip()) < 100:
                from pdf2image import convert_from_path
                import easyocr

                reader = easyocr.Reader(['en'], gpu=False)
                images = convert_from_path(file_path)
                ocr_text = "\n".join([reader.readtext(img, detail=0, paragraph=True)[0] for img in images])
                return ocr_text[:max_chars]
            return text[:max_chars]
        except Exception as e:
            raise ValueError(f"PDF extraction failed: {str(e)}")

    elif ext == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()[:max_chars]

    elif ext in ["doc", "docx"]:
        import docx2txt
        return docx2txt.process(file_path)[:max_chars]

    elif ext in ["xls", "xlsx"]:
        import pandas as pd
        df = pd.read_excel(file_path, dtype=str)
        return df.to_string(index=False)[:max_chars]

    else:
        raise ValueError(f"Unsupported file format: {ext}")

def build_prompt(text):
    prompt = f"""
You are an agreement extraction assistant. Given the following document text, extract the required parameters under each category. For each parameter, return a JSON object with its category, parameter name, and value. If not found, return null.

Document content:
{text}

Parameters to extract:
"""
    for category, fields in parameter_categories.items():
        prompt += f"\n\n{category}:"
        for param, desc in fields.items():
            prompt += f"\n- {param}: {desc}"

    prompt += """

Return JSON in the following format:
[
  {"category": "Borrower Details", "parameter": "Name of the Borrower (Legal Name)", "value": "..."},
  ...
]
"""
    return prompt

def clean_llm_output(raw_output):
    import re
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_output, re.DOTALL)
    return match.group(1).strip() if match else raw_output.strip()

def query_together(prompt):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": TOGETHER_MODEL,
        "messages": [
            {"role": "system", "content": "You are an agreement extraction assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 2048
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=body,
            verify=certifi.where(),
            timeout=30
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return json.dumps({
            "error": f"LLM error: {str(e)}",
            "llm_output": None
        })

@app.route("/extract-fields", methods=["POST"])
def extract_fields():
    files = request.files.getlist("file")
    if not files:
        return jsonify({"error": "No files provided"}), 400

    results = []

    for file in files:
        if file.filename == "":
            continue

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1]) as tmp:
                file.save(tmp.name)
                tmp_path = tmp.name

            raw_text = extract_text(tmp_path)
            prompt = build_prompt(raw_text)
            llm_output = query_together(prompt)
            cleaned_output = clean_llm_output(llm_output)

            try:
                parsed = json.loads(cleaned_output)
            except json.JSONDecodeError:
                parsed = {
                    "llm_raw": llm_output,
                    "error": "Invalid JSON from LLM"
                }

            results.append({"filename": file.filename, "extracted_fields": parsed})

        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return jsonify(results)

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
