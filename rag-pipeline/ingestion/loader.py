import os
import re
import json

# ---------------- PDF Loaders ---------------- #

try:
    from langchain_community.document_loaders import UnstructuredPDFLoader
except Exception:
    UnstructuredPDFLoader = None

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


# ---------------- Cleaning ---------------- #

def clean_text(text: str) -> str:
    """
    Basic normalization for PDFs and TXT files.
    Keeps content while removing obvious extraction noise.
    """

    if not text:
        return ""

    # normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # remove excessive spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    # remove common page markers
    text = re.sub(
        r"Page\s+\d+(\s+of\s+\d+)?",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = text.encode("utf-8", errors="ignore").decode("utf-8")

    return text.strip()


# ---------------- TXT Loader ---------------- #

def load_txt_file(file_path, file_name, category):

    documents = []

    try:
        with open(
            file_path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:

            text = clean_text(f.read())

        if text:

            documents.append({
                "text": text,
                "source": file_name,
                "title": os.path.splitext(file_name)[0],
                "source_path": file_path,
                "category": category,
                "doc_type": "txt",
                "page": None
            })

    except Exception as e:
        print(f"[TXT ERROR] {file_name}: {e}")

    return documents


# # ---------------- Unstructured Loader ---------------- #

# def load_pdf_unstructured(file_path, file_name, category):

#     documents = []

#     if UnstructuredPDFLoader is None:
#         return documents

#     try:

#         loader = UnstructuredPDFLoader(
#             file_path=file_path,
#             mode="elements"
#         )

#         elements = loader.load()

#         for idx, element in enumerate(elements, start=1):

#             page_text = clean_text(
#                 getattr(element, "page_content", "")
#             )

#             metadata = getattr(
#                 element,
#                 "metadata",
#                 {}
#             ) or {}

#             page_num = (
#                 metadata.get("page_number")
#                 or metadata.get("page")
#                 or idx
#             )

#             if len(page_text) < 30:
#                 continue

#             documents.append({
#                 "text": page_text,
#                 "source": file_name,
#                 "title": os.path.splitext(file_name)[0],
#                 "source_path": file_path,
#                 "category": category,
#                 "doc_type": "pdf",
#                 "page": page_num
#             })

#         if documents:
#             print(f"[UNSTRUCTURED] {file_name}")

#         return documents

#     except Exception as e:

#         print(
#             f"[UNSTRUCTURED FAILED] "
#             f"{file_name}: {e}"
#         )

#         return []


# ---------------- PyMuPDF Loader ---------------- #

def load_pdf_pymupdf(file_path, file_name, category):

    documents = []

    if fitz is None:
        return documents

    try:

        doc = fitz.open(file_path)

        for page_num in range(len(doc)):

            page = doc[page_num]

            page_text = page.get_text("text")

            if not isinstance(page_text, str):
                page_text = ""

            page_text = clean_text(page_text)

            if len(page_text) < 30:
                continue

            documents.append({
                "text": page_text,
                "source": file_name,
                "title": os.path.splitext(file_name)[0],
                "source_path": file_path,
                "category": category,
                "doc_type": "pdf",
                "page": page_num + 1
            })

        if documents:
            print(f"[PYMUPDF] {file_name}")

        return documents

    except Exception as e:

        print(
            f"[PYMUPDF FAILED] "
            f"{file_name}: {e}"
        )

        return []


# ---------------- PyPDF Loader ---------------- #

def load_pdf_pypdf(file_path, file_name, category):

    documents = []

    if PdfReader is None:
        return documents

    try:

        reader = PdfReader(file_path)

        for page_num, page in enumerate(
            reader.pages,
            start=1
        ):

            page_text = clean_text(
                page.extract_text() or ""
            )

            if len(page_text) < 30:
                continue

            documents.append({
                "text": page_text,
                "source": file_name,
                "title": os.path.splitext(file_name)[0],
                "source_path": file_path,
                "category": category,
                "doc_type": "pdf",
                "page": page_num
            })

        if documents:
            print(f"[PYPDF] {file_name}")

        return documents

    except Exception as e:

        print(
            f"[PYPDF FAILED] "
            f"{file_name}: {e}"
        )

        return []


# ---------------- Main Loader ---------------- #

def load_documents(data_path="data/raw"):

    documents = []

    for root, _, files in os.walk(data_path):

        category = os.path.basename(root)

        for file_name in files:

            file_path = os.path.join(
                root,
                file_name
            )

            # ---------- TXT ---------- #

            if file_name.lower().endswith(".txt"):

                documents.extend(
                    load_txt_file(
                        file_path,
                        file_name,
                        category
                    )
                )

            # ---------- PDF ---------- #

            elif file_name.lower().endswith(".pdf"):

                pdf_docs = load_pdf_pymupdf(
                    file_path,
                    file_name,
                    category
                )

                if not pdf_docs:

                    pdf_docs = load_pdf_pypdf(
                        file_path,
                        file_name,
                        category
                    )

                documents.extend(pdf_docs)

    return documents


# ---------------- Debug ---------------- #

if __name__ == "__main__":

    project_root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            ".."
        )
    )

    data_path = os.path.join(
        project_root,
        "data",
        "raw"
    )

    docs = load_documents(data_path)

    print("\n" + "=" * 60)
    print(f"Loaded {len(docs)} documents")
    print("=" * 60)

    if docs:
        print(
            json.dumps(
                docs[0],
                indent=2,
                ensure_ascii=False
            )
        )