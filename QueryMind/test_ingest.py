import fitz
import urllib.request
import json

doc = fitz.open()
page = doc.new_page()
page.insert_text((50, 50), "Contract Supplier A. Expiry: 2025-12-31.")
doc.save("/tmp/test.pdf")
doc.close()

boundary = "boundary123"
with open("/tmp/test.pdf", "rb") as f:
    pdf_data = f.read()

body = (
    b"--boundary123\r\n"
    b"Content-Disposition: form-data; name=\"file\"; filename=\"test.pdf\"\r\n"
    b"Content-Type: application/pdf\r\n\r\n"
    + pdf_data
    + b"\r\n--boundary123--\r\n"
)

req = urllib.request.Request(
    "http://localhost:8000/rag/ingest",
    data=body,
    headers={"Content-Type": "multipart/form-data; boundary=boundary123"},
    method="POST"
)

res = urllib.request.urlopen(req)
print(json.loads(res.read()))
