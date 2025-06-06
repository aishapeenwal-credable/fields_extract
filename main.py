import os
import re
import ssl
import json
import pdfplumber
import tempfile
import together
import requests
from urllib3.poolmanager import PoolManager
from requests.adapters import HTTPAdapter
from fastapi import FastAPI, File, UploadFile, Form
from typing import List, Optional
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ---------- Unsafe SSL Adapter ----------
class UnsafeAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)

s = requests.Session()
s.mount("https://", UnsafeAdapter())
requests.get = s.get
requests.post = s.post
requests.request = s.request
requests.head = s.head
requests.put = s.put
requests.delete = s.delete

# ---------- FastAPI ----------
app = FastAPI()

# ---------- CORS Settings ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://lovable-ui.vercel.app",
        "https://id-preview--c7ef1364-e7e3-4010-929a-ace0b3c13062.lovable.app",
        "https://c7ef1364-e7e3-4010-929a-ace0b3c13062.lovable.app",
        "https://c7ef1364-e7e3-4010-929a-ace0b3c13062.lovableproject.com",
        "https://preview--agreement-generation.lovable.app",
        "https://agreement-generation.lovable.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
) 

together.api_key = os.getenv("TOGETHER_API_KEY")


together.api_key = os.getenv("TOGETHER_API_KEY")

# --- Keep all your existing imports and unsafe SSL patch code ---

# ---------- FastAPI ----------
app = FastAPI()
together.api_key = os.getenv("TOGETHER_API_KEY")
# ---------- Flat parameter list ----------
parameter_fields = [
    "doc_deed_of_hypothecation_applicable(true_or_false)", "doc_cover_letter_applicable(true_or_false)", "doc_deed_of_personal_guarantee_applicable(true_or_false)", "doc_deed_of_corporate_guarantee_applicable(true_or_false)", "doc_undertaking_applicable(true_or_false)",
    "borrower_name", "borrower_constitution", "borrower_cin", "borrower_pan", "borrower_registered_address", "director_name", "director_address",
    "facility_amount", "facility_amount_in_words", "interest_rate", "tenor", "cure_period", "default_charges",
    "maximum_disbursement", "validity", "platform_service_fee", "transaction_fee", "minimum_utilisation", "pari_pasu_applicable(true_or_false)",
    "Pari_pasu_charge", "fldg_applicable(true_or_false)", "conditions_precedent", "conditions_subsequent", "finance_documents", "end_clients", "security"
]

def build_prompt(text: str) -> str:
    schema_fields = "\n".join(f"- {field}" for field in parameter_fields)
    return f"""You are an expert assistant extracting structured data from loan sanction documents.

Below is the text extracted from a PDF credit appraisal note. Extract values ONLY for the fields listed in the schema.
If not identifiable, return \"Null\".

Return the output as a list of JSON objects in the format: 
[{{\"Field\": \"<field_name>\", \"Value\": \"<value>\"}}]

### Document Text:
{text}

### Schema Fields:
{schema_fields}

### Output JSON:
"""

def call_together_llm(prompt: str) -> str:
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")
    input_tokens = len(encoding.encode(prompt))
    max_tokens = max(256, 8192 - input_tokens)

    try:
        response = together.Complete.create(
            model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.2,
            stop=["###"]
        )

        if "choices" in response and len(response["choices"]) > 0:
            raw_text = response["choices"][0]["text"].strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[len("```json"):].strip()
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3].strip()
            return raw_text
        return "[]"

    except Exception as e:
        return json.dumps([{"Field": "exception", "Value": str(e)}])

def find_page_and_excerpt(value: str, pages: list) -> tuple:
    for i, page_text in enumerate(pages):
        if value and isinstance(value, str) and value.strip().lower() not in {"null", ""}:
            pattern = re.escape(value.strip()[:20])
            if re.search(pattern, page_text, flags=re.IGNORECASE):
                excerpt = page_text[page_text.lower().find(value.lower()[:20]):][:150]
                return i + 1, excerpt
    return None, None

def extract_fields_from_pdf_llm(pdf_path: str, source_name: str) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        pages_text = [page.extract_text() or "" for page in pdf.pages]
        full_text = "\n".join(pages_text)

    trimmed_text = full_text[:12000]
    prompt = build_prompt(trimmed_text)
    raw_response = call_together_llm(prompt)

    try:
        json_blocks = re.findall(r"\[\s*{.*?}\s*\]", raw_response, flags=re.DOTALL)
        merged_items = []
        for block in json_blocks:
            try:
                parsed = json.loads(block)
                for item in parsed:
                    val = item.get("Value", "")
                    page_num, excerpt = find_page_and_excerpt(val, pages_text)
                    item["SourceDocument"] = source_name
                    item["PageNumber"] = page_num if page_num else "Unknown"
                    item["Excerpt"] = excerpt if excerpt else "N/A"
                merged_items.extend(parsed)
            except json.JSONDecodeError:
                continue
        return json.dumps(merged_items, indent=2)

    except Exception as e:
        return json.dumps([{
            "Field": "error",
            "Value": f"LLM returned malformed JSON: {str(e)}",
            "SourceDocument": source_name
        }], indent=2)

@app.post("/extract-fields")
async def extract_fields(
    files: List[UploadFile] = File(...),
    priority: Optional[str] = Form(None)
):
    def get_priority_index(filename):
        return 0 if filename == priority else 1

    field_map = {}

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name

        result_str = extract_fields_from_pdf_llm(tmp_path, source_name=file.filename)
        result_items = json.loads(result_str)

        for item in result_items:
            key = item["Field"]
            new_entry = {
                "Value": item["Value"],
                "SourceDocument": item["SourceDocument"],
                "PageNumber": item["PageNumber"],
                "Excerpt": item["Excerpt"],
                "Priority": get_priority_index(item["SourceDocument"])
            }

            if key not in field_map:
                field_map[key] = {
                    "Field": key,
                    "Value": item["Value"],
                    "SourceDocument": item["SourceDocument"],
                    "PageNumber": item["PageNumber"],
                    "Excerpt": item["Excerpt"],
                    "Priority": new_entry["Priority"],
                    "AlternateValues": [],
                    "Conflicting": False
                }
            else:
                existing = field_map[key]
                if item["Value"] == existing["Value"]:
                    continue
                if new_entry["Priority"] < existing["Priority"]:
                    existing["AlternateValues"].append({
                        "Value": existing["Value"],
                        "SourceDocument": existing["SourceDocument"],
                        "PageNumber": existing["PageNumber"],
                        "Excerpt": existing["Excerpt"]
                    })
                    existing.update({
                        "Value": item["Value"],
                        "SourceDocument": item["SourceDocument"],
                        "PageNumber": item["PageNumber"],
                        "Excerpt": item["Excerpt"],
                        "Priority": new_entry["Priority"],
                        "Conflicting": True
                    })
                else:
                    existing["AlternateValues"].append(new_entry)
                    existing["Conflicting"] = True

    output = {}
    for entry in field_map.values():
        entry.pop("Priority")
        output[entry["Field"]] = entry
        entry.pop("Field")

    return JSONResponse(content=output)
