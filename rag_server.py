from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings
)
from llama_index.llms.openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding

import uvicorn
from fastapi import FastAPI

Settings.node_parser = SentenceSplitter(chunk_size=300, chunk_overlap=50)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")


Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)


doc = SimpleDirectoryReader(
    r"C:\Users\USER\Desktop\books\policy",
    required_exts=[".pdf"]
).load_data()

index = VectorStoreIndex.from_documents(doc)
query_engine = index.as_query_engine()

app = FastAPI()

@app.post("/rag_credit")
def credit_rag(payload:dict):
    prompt = f"""
    You are a Credit QA analyst.

    Loan PD: {payload['PD']:.2%}
    Risk Band: {payload['risk_band']}
    Income: {payload['income']}
    DTI: {payload['dti']}%
    Loan Amount: {payload['loan_amount']}
    Interest Rate: {payload['interest_rate']}%

    Using the bank's credit policy and market outlook, explain:
    1. Why this loan is risky or safe
    2. Which policy rules apply
    3. What the recommended action should be
    """

    response = query_engine.query(prompt)
    return {"recommendation": str(response)}
