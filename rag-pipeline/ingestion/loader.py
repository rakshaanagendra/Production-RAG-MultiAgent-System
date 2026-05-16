import os
import json

# Try both PDF loaders (safe fallback)
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from langchain_community.document_loaders import UnstructuredPDFLoader
except Exception:
    UnstructuredPDFLoader = None


def load_documents(data_path="data/raw"):
    documents = []

    for root, _, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root, file)

            # ---------------- TXT ----------------
            if file.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()

                if text:
                    documents.append({
                        "text": text,
                        "source": file,
                        "source_path": file_path,
                        "doc_type": "txt",
                        "page": None
                    })

            # ---------------- PDF ----------------
            elif file.endswith(".pdf"):
                if UnstructuredPDFLoader is not None:
                    try:
                        # Use element mode to preserve richer PDF structure and metadata.
                        loader = UnstructuredPDFLoader(file_path=file_path, mode="elements")
                        elements = loader.load()

                        for idx, element in enumerate(elements, start=1):
                            page_text = (getattr(element, "page_content", "") or "").strip()
                            metadata = getattr(element, "metadata", {}) or {}
                            page_num = metadata.get("page_number") or metadata.get("page") or idx

                            if page_text:
                                documents.append({
                                    "text": page_text,
                                    "source": file,
                                    "source_path": file_path,
                                    "doc_type": "pdf",
                                    "page": page_num
                                })
                        continue
                    except Exception:
                        # Fall back to local PDF parsers if unstructured is installed but fails.
                        pass

                if fitz is not None:
                    doc = fitz.open(file_path)

                    for page_num in range(len(doc)):
                        page = doc[page_num]
                        page_text = page.get_text()

                        if not isinstance(page_text, str):
                            page_text = ""

                        page_text = page_text.strip()

                        if page_text:
                            documents.append({
                                "text": page_text,
                                "source": file,
                                "source_path": file_path,
                                "doc_type": "pdf",
                                "page": page_num
                            })

                elif PdfReader is not None:
                    reader = PdfReader(file_path)

                    for page_num, page in enumerate(reader.pages, start=1):
                        page_text = page.extract_text() or ""
                        page_text = page_text.strip()

                        if page_text:
                            documents.append({
                                "text": page_text,
                                "source": file,
                                "source_path": file_path,
                                "doc_type": "pdf",
                                "page": page_num
                            })

    return documents


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    resolved_data_path = os.path.join(project_root, "data", "raw")

    documents = load_documents(resolved_data_path)
    print(f"Loaded {len(documents)} document(s) from: {resolved_data_path}")

    if UnstructuredPDFLoader is None and fitz is None and PdfReader is None:
        print("PDF parsing is disabled. Install unstructured, PyMuPDF, or pypdf to include .pdf files.")

    if not documents:
        print("No loadable .txt/.pdf files were found.")
    else:
        for doc in documents[:5]:
            print(json.dumps(doc, ensure_ascii=False, indent=2))