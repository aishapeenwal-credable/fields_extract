import os
import tempfile
import pandas as pd
import pytesseract
import certifi
import json
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from pdf2image import convert_from_bytes
from PyPDF2 import PdfReader
from PIL import Image
from dotenv import load_dotenv
import requests

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["*", "https://lovable.so"]}})

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_MODEL = os.getenv("TOGETHER_LLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")

parameter_categories = {
    "Borrower Details": {
        "Name of the Borrower (Legal Name)": "The official legal name of the entity taking the loan.",
        "Constitution": "The type of legal entity (e.g., Pvt Ltd, LLP, Partnership).",
        "CIN": "Corporate Identification Number of the borrowing entity.",
        "PAN": "Permanent Account Number of the borrower.",
        "Registered Address": "Official address registered with authorities.",
        "Name of the Director": "Full name of the director involved.",
        "Address of the Director": "Residential address of the director."
    },
    "Sanction Details": {
        "Facility / Loan Amount": "Total sanctioned loan or facility amount.",
        "Facility Agreement Date": "Date on which the facility agreement was executed.",
        "Interest": "Applicable interest rate on the loan.",
        "Tenor": "Duration or term of the facility.",
        "Cure Period": "Time provided to rectify a default before action.",
        "Default Charges": "Penalties for default on repayment.",
        "Maximum disbursement": "Upper cap of loan that can be disbursed.",
        "Validity": "Period for which the sanction letter or offer is valid.",
        "Platform Service Fee": "Fee charged by the platform managing the loan.",
        "Transaction Fee": "Fee for processing the loan transaction.",
        "Minimum Utilisation": "Minimum amount of facility that must be used.",
        "Pari-pasu applicable (Yes or no)": "Whether the lender shares rights equally with others.",
        "FLDG (applicable Yes or no)": "First Loss Default Guarantee applicability.",
        "Conditions Precedent": "Conditions that must be fulfilled before disbursement.",
        "Conditions Subsequent": "Conditions to be fulfilled after disbursement.",
        "Finance Documents": "All legal documents related to the financing."
    },
    "Individual Guarantor Details": {
        "Individual Name": "Full name of the individual guarantor.",
        "Age of Indivudual Guarantor": "Age of the individual providing the guarantee.",
        "Age of the guarantor": "Duplicate of the above (keep consistent).",
        "Guarantor’s PAN": "Permanent Account Number of the guarantor.",
        "Guarantor’s father’s name": "Father’s name of the individual guarantor.",
        "Father’s Name": "Same as above (duplicate label).",
        "Residential Address": "Guarantor’s home address.",
        "Contact details (name, email, phone) of both parties": "Contact details for borrower and lender representatives.",
        "Guarantor Name": "Name of the individual guarantor."
    },
    "Corporate Guarantor Details": {
        "Guaranteed %": "Percentage of the loan amount guaranteed.",
        "Email ID of Guarantor": "Email address of the corporate guarantor.",
        "CIN": "Corporate Identification Number of the corporate guarantor.",
        "Business Address": "Business address of the corporate guarantor."
    },
    "Notice details of the Lender": {
        "Name (Attention)": "Mr. Ketan Mehta",
        "Address": "5th Floor, Satyam Tower, Off Govandi Station Road, Deonar, Mumbai-400088",
        "Email": "finance2@credable.in",
        "Contact number": "022-49266964 / 49266900"
    },
    "Notice details of the Borrower": {
        "Name (Attention)": "Contact person at the borrower’s end.",
        "Address": "Mailing address for borrower legal notices.",
        "Email": "Email address for borrower notices.",
        "Contact number": "Phone number for borrower notices."
    },
    "Authorised Signatory Details": {
        "Name": "Name of the person authorized to sign.",
        "PAN (Image)": "PAN card image of the authorized signatory.",
        "AADHAR (image)": "Aadhaar card image of the authorized signatory."
    }
}

def extract_text(file_path):
    ext = file_path.lower().split('.')[-1]
    
    if ext == "pdf":
        with open(file_path, 'rb') as f:
            pdf_bytes = f.read()
        
        reader = PdfReader(io.BytesIO(pdf_bytes))
        total_pages = len(reader.pages)

        extracted_text = []

        for page_num in range(1, total_pages + 1):
            images = convert_from_bytes(
                pdf_bytes, dpi=100, first_page=page_num, last_page=page_num
            )
            for img in images:
                text = pytesseract.image_to_string(img)
                extracted_text.append(text)
                del img  # free memory

        return "\n".join(extracted_text)

    elif ext in ["xlsx", "xls"]:
        df = pd.read_excel(file_path)
        return df.to_string(index=False)

    elif ext in ["jpg", "jpeg", "png"]:
        image = Image.open(file_path)
        return pytesseract.image_to_string(image)

    elif ext == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    else:
        raise ValueError(f"Unsupported file format: {ext}")

def build_prompt(text):
    prompt = f"""
You are an agreement extraction assistant. Given the following document text, extract the required parameters under each category. For each parameter, return a JSON object with its category, parameter name, and value. If the parameter is not found, return null.

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

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1]) as tmp:
        file.save(tmp.name)
        try:
            raw_text = extract_text(tmp.name)
            prompt = build_prompt(raw_text)
            llm_output = query_together(prompt)
            try:
                parsed = json.loads(llm_output)
            except json.JSONDecodeError:
                parsed = llm_output
            return jsonify({"extracted_fields": parsed})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            os.unlink(tmp.name)

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
