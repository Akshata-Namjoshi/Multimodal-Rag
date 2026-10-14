# Multimodal RAG for PDFs

A local Streamlit app for searching PDF text, extracting tables, inspecting page images, and asking questions about a selected page. The project explores how retrieval-augmented generation can use both document text and visual evidence.

## Problem

Text-only PDF search can miss information in tables, diagrams, charts, and screenshots. This app combines extracted page text and tables with a rendered page image so a local vision-language model can describe visual evidence and a local language model can form an answer.

## Features

- Upload a PDF in the browser.
- Extract page text with `pypdf`.
- Extract table rows with `pdfplumber`; view detected tables in the app.
- Search extracted text and table content using TF-IDF and cosine similarity.
- Render complete PDF pages with `pypdfium2` and show embedded image objects where available.
- Ask a question about a selected page. The app retrieves text and tables from that page, sends its rendered image to Moondream for a visual summary, and asks Qwen 3 to answer using the combined evidence.
- Run inference through a local Ollama server. No paid API key is required.

## Architecture

```text
PDF upload
  ├── pypdf ───────────────> page text ──> chunks ──> TF-IDF search
  ├── pdfplumber ──────────> table rows ────────────> table view / page context
  └── pypdfium2 ───────────> rendered page image ──> Moondream visual summary
                                                       │
Selected-page text + tables + visual summary ─────────┴─> Qwen 3 answer
```

The current text retriever is a lightweight TF-IDF baseline. It is useful for a free, local prototype, but it is not a dense-vector or multi-vector retriever.

## Technology

- Python
- Streamlit
- pypdf
- pdfplumber
- pypdfium2 and Pillow
- scikit-learn (TF-IDF and cosine similarity)
- Ollama with `moondream` and `qwen3:0.6b`

## Run locally on Windows

Open PowerShell in the project folder:

```powershell
cd D:\Akshata\Multimodal-Rag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Install the local models once:

```powershell
ollama pull moondream
ollama pull qwen3:0.6b
```

Make sure Ollama is running, then start the app:

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit, upload a PDF, and use the **Text search**, **Tables**, or **Page previews and Q&A** tab.

## Evaluation status

Manual checks so far:

- On `2610.10544v1.pdf`, the Table I question “According to Table I, how much memory does the edge gateway have?” returned **4 GB LPDDR4-3200 SDRAM** in both Text search and page Q&A.
- A visual question about Figure 3 has not yet been verified in the page Q&A flow. A result showing ranked matches and `match:` scores is from Text search; it does not demonstrate image understanding.
- No overall accuracy, recall, or citation-accuracy score is reported yet. These should be calculated only after running a documented set of questions and recording the actual outputs.

## Current limitations

- PDF table extraction depends on the document layout; scanned pages and irregular tables may need OCR or manual review.
- Text retrieval is lexical TF-IDF, so it can miss paraphrases and may rank pages with overlapping words.
- The visual model can misread small text or diagram labels. Check answers against the selected page and its evidence.
- The app currently uses a rendered full page as visual input; it does not crop individual figures or charts.
- Answers depend on the local models installed in Ollama and available computer memory.

## Repository contents

```text
Multimodal-Rag/
├── app.py
├── requirements.txt
├── data/
├── pdf/
└── README.md
```

Keep documents used for testing in a suitable local folder. Only commit PDFs if you have the right to redistribute them and they do not contain private information.

## Next steps

1. Verify page-specific visual Q&A on Figure 3 and inspect the app's **Evidence used** section.
2. Test table extraction and retrieval on several PDFs with different layouts.
3. Record questions, expected answers, actual answers, and cited pages before calculating evaluation metrics.
4. Consider replacing TF-IDF with a local embedding retriever and adding a multi-vector index that links image summaries back to their original pages or images.
