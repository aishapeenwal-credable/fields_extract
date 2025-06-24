import os
import re
import ssl
import json
import time
import pdfplumber
import tempfile
import together
import requests
import tiktoken
from urllib3.poolmanager import PoolManager
from requests.adapters import HTTPAdapter
from fastapi import FastAPI, File, UploadFile, Form
from typing import List, Optional
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ---------- SSL Bypass + User-Agent ----------
class UnsafeAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)

session = requests.Session()
session.mount("https://", UnsafeAdapter())
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/113.0.0.0 Safari/537.36"
})

# ---------- FastAPI Setup ----------
app = FastAPI()

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

# ---------- Config ----------
together.api_key = os.getenv("TOGETHER_API_KEY")

parameter_fields = [
    "doc_deed_of_hypothecation_applicable(true_or_false)", "doc_cover_letter_applicable(true_or_false)",
    "doc_deed_of_personal_guarantee_applicable(true_or_false)", "doc_deed_of_corporate_guarantee_applicable(true_or_false)",
    "borrower_name", "borrower_constitution", "borrower_cin", "borrower_pan",
    "borrower_registered_address", "director_name", "director_address", "facility_amount",
    "facility_amount_in_words", "interest_rate", "tenor", "cure_period", "default_charges",
    "maximum_disbursement", "validity", "platform_service_fee", "transaction_fee",
    "minimum_utilisation", "pari_pasu_applicable(true_or_false)", "Pari_pasu_charge",
    "fldg_applicable(true_or_false)", "conditions_precedent", "conditions_subsequent",
    "finance_documents", "end_clients", "security"
]

# ---------- Health Check Utilities ----------
def check_llm_health() -> None:
    try:
        together.Complete.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            prompt="ping",
            max_tokens=1,
            temperature=0.0
        )
    except Exception as e:
        raise RuntimeError(f"LLM API is not reachable (health check failed): {e}")

# ---------- Network Retry Utility ----------
def call_with_retry(fn, retries=3, base_delay=2):
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt == retries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))

# ---------- Prompt & Token Utilities ----------
def safe_trim(text: str, max_tokens: int = 6000):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    return enc.decode(tokens[:max_tokens])

def build_prompt(text: str) -> str:
    formatted_fields = "\n".join(
        f"- {f.replace('(true_or_false)', '')} (return 'true' or 'false')" if "(true_or_false)" in f else f"- {f}"
        for f in parameter_fields
    )
    return f"""You are an expert assistant extracting structured data from loan sanction documents.

Below is the text extracted from a PDF credit appraisal note. Extract values ONLY for the fields listed in the schema.
If not identifiable, return "Null".
If a field ends with (true_or_false), only return "true" or "false" as lowercase strings.
Return the output as a list of JSON objects in the format:
[{{"Field": "<field_name>", "Value": "<value>"}}]

### Document Text:
{text}

### Schema Fields:
{formatted_fields}

### Output JSON:
"""

def call_together_llm(prompt: str) -> str:
    def make_request():
        encoding = tiktoken.get_encoding("cl100k_base")
        input_tokens = len(encoding.encode(prompt))
        max_tokens = max(256, 8192 - input_tokens)
        response = together.Complete.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.2,
            stop=["###"]
        )
        raw_text = response["choices"][0]["text"].strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[len("```json"):].strip()
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3].strip()
        return raw_text

    try:
        return call_with_retry(make_request)
    except Exception as e:
        return json.dumps([{
            "Field": "exception",
            "ErrorType": e.__class__.__name__,
            "Message": str(e)
        }])

def find_page_and_excerpt(value: str, pages: list) -> tuple:
    for i, page_text in enumerate(pages):
        if value and isinstance(value, str) and value.strip().lower() not in {"null", ""}:
            pattern = re.escape(value.strip()[:20])
            if re.search(pattern, page_text, flags=re.IGNORECASE):
                excerpt = page_text[page_text.lower().find(value.lower()[:20]):][:150]
                return i + 1, excerpt
    return None, None

def get_applicability_booleans(text: str, security_text: str) -> dict:
    text_lower = text.lower()
    sec_lower = security_text.lower()
    return {
        "doc_deed_of_hypothecation_applicable": "true" if "hypothecation" in text_lower else "false",
        "doc_cover_letter_applicable": "true" if "cheque" in sec_lower else "false",
        "doc_deed_of_personal_guarantee_applicable": "true" if "personal guarantee" in sec_lower or "pg" in sec_lower else "false",
        "doc_deed_of_corporate_guarantee_applicable": "true" if "corporate guarantee" in sec_lower or "cg" in sec_lower else "false",
    }

# ---------- Health Route ----------
@app.get("/health")
def health_check():
    try:
        check_llm_health()
        return {"status": "ok", "llm_api": True}
    except Exception as e:
        return {"status": "unhealthy", "llm_api": False, "error": str(e)}

# ---------- Main Endpoint ----------
@app.post("/extract-fields")
async def extract_fields(
    files: List[UploadFile] = File(...),
    priority: Optional[str] = Form(None)
):
    check_llm_health()

    def get_priority_index(filename):
        return 0 if filename == priority else 1

    field_map = {}

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name

        with pdfplumber.open(tmp_path) as pdf:
            pages_text = [page.extract_text() or "" for page in pdf.pages]
            full_text = "\n".join(pages_text)

        prompt = build_prompt(safe_trim(full_text))
        raw_response = call_together_llm(prompt)

        try:
            result_items = json.loads(raw_response)
            if not isinstance(result_items, list):
                raise ValueError("Top-level JSON is not a list")
        except:
            result_items = []
            for block in re.findall(r"\[\s*{.*?}\s*\]", raw_response, flags=re.DOTALL):
                try:
                    result_items.extend(json.loads(block))
                except:
                    continue

        for item in result_items:
            val = item.get("Value", "")
            page_num, excerpt = find_page_and_excerpt(val, pages_text)
            item.update({
                "SourceDocument": file.filename,
                "PageNumber": page_num or "Unknown",
                "Excerpt": excerpt or "N/A",
                "FieldPresent": val.lower() != "null" and val != ""
            })

        security_val = next((i.get("Value", "") for i in result_items if i.get("Field") == "security"), "")
        for key, val in get_applicability_booleans(full_text, security_val).items():
            result_items.append({
                "Field": key,
                "Value": val,
                "SourceDocument": file.filename,
                "PageNumber": "Auto",
                "Excerpt": "Derived from text analysis.",
                "FieldPresent": True
            })

        for item in result_items:
            key = item["Field"]
            new_entry = {
                "Value": item["Value"],
                "SourceDocument": item["SourceDocument"],
                "PageNumber": item["PageNumber"],
                "Excerpt": item["Excerpt"],
                "Priority": get_priority_index(item["SourceDocument"]),
                "FieldPresent": item["FieldPresent"]
            }

            if key not in field_map:
                field_map[key] = {
                    "Field": key,
                    **new_entry,
                    "AlternateValues": [],
                    "Conflicting": False
                }
            else:
                existing = field_map[key]
                if item["Value"] == existing["Value"]:
                    continue
                if new_entry["Priority"] < existing["Priority"]:
                    existing["AlternateValues"].append({
                        k: existing[k] for k in ["Value", "SourceDocument", "PageNumber", "Excerpt"]
                    })
                    existing.update(new_entry)
                    existing["Conflicting"] = True
                else:
                    existing["AlternateValues"].append(new_entry)
                    existing["Conflicting"] = True

    output = {}
    for entry in field_map.values():
        entry.pop("Priority")
        output[entry["Field"]] = entry
        entry.pop("Field")

    return JSONResponse(content=output)
