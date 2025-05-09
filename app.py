import os
import tempfile
import certifi
import json
import pdfplumber
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["*", "https://lovable.so"]}})

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_MODEL = os.getenv("TOGETHER_LLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")

# Load parameters from external config
from parameter_config import parameter_categories


def extract_text(file_path):
    ext = file_path.lower().split('.')[-1]

    if ext == "pdf":
        try:
            with pdfplumber.open(file_path) as pdf:
                extracted_text = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        extracted_text.append(page_text.strip())
            return "\n".join(extracted_text)
        except Exception as e:
            raise ValueError(f"Failed to extract from PDF: {str(e)}")

    elif ext == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    else:
        raise ValueError(f"Unsupported file format: {ext}")


def build_prompt(text):
    text = text[:8000]  # truncate to stay within LLM token limits
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
            verify=certifi.where()
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return json.dumps({
            "document_type": "Unknown",
            "confidence": 0.0,
            "reason": f"LLM error: {str(e)}"
        })


@app.route("/extract-fields", methods=["POST"])
def extract_fields():
    if "file" not in request.files:
        return jsonify({"error": "File not provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Filename is empty"}), 400

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1]) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        raw_text = extract_text(tmp_path)
        prompt = build_prompt(raw_text)
        llm_output = query_together(prompt)

        try:
            parsed = json.loads(llm_output)
        except json.JSONDecodeError:
            parsed = {"llm_raw": llm_output}

        return jsonify({"extracted_fields": parsed})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
