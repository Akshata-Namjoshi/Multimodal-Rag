from io import BytesIO
import base64

import pdfplumber
import pypdfium2 as pdfium
import requests
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_text_pages(pdf_bytes: bytes, filename: str) -> list[dict]:
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []

    # Open a fresh BytesIO because PdfReader has already read the original stream.
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for page_index, pdf_page in enumerate(reader.pages):
            page_number = page_index + 1
            parts = []

            # Extract regular page text.
            page_text = (pdf_page.extract_text() or "").strip()
            if page_text:
                parts.append(page_text)

            # Extract tables and turn each row into searchable text.
            if page_index < len(pdf.pages):
                tables = pdf.pages[page_index].extract_tables()

                for table_number, table in enumerate(tables, start=1):
                    rows = []

                    for row in table:
                        cells = [
                            (cell or "").replace("\n", " ").strip()
                            for cell in row
                        ]

                        if any(cells):
                            rows.append(" | ".join(cells))

                    if rows:
                        parts.append(
                            f"TABLE {table_number}:\n" + "\n".join(rows)
                        )

            # Keep pages that contain either text or a table.
            combined_text = "\n\n".join(parts).strip()
            if combined_text:
                pages.append(
                    {
                        "source": filename,
                        "page": page_number,
                        "text": combined_text,
                    }
                )

    return pages


def extract_tables(pdf_bytes: bytes) -> list[dict]:
    """Try ruled-line and text-alignment strategies to find PDF tables."""
    tables = []

    strategies = [
        None,  # Default: detect cell borders from PDF lines.
        {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "min_words_vertical": 2,
            "min_words_horizontal": 1,
            "text_x_tolerance": 5,
            "text_y_tolerance": 5,
        },
        {
            "vertical_strategy": "lines",
            "horizontal_strategy": "text",
            "min_words_horizontal": 1,
            "text_y_tolerance": 5,
        },
    ]

    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for page_number, pdf_page in enumerate(pdf.pages, start=1):
            best_tables = []
            best_score = 0

            for settings in strategies:
                if settings is None:
                    found = pdf_page.extract_tables()
                else:
                    found = pdf_page.extract_tables(table_settings=settings)

                found = found or []

                valid_tables = [
                    table
                    for table in found
                    if table
                    and len(table) >= 2
                    and max((len(row) for row in table), default=0) >= 2
                ]

                score = sum(
                    1
                    for table in valid_tables
                    for row in table
                    for cell in row
                    if cell and str(cell).strip()
                )

                if score > best_score:
                    best_tables = valid_tables
                    best_score = score

            for table_number, rows in enumerate(best_tables, start=1):
                tables.append(
                    {
                        "page": page_number,
                        "table_number": table_number,
                        "rows": rows,
                    }
                )

    return tables


def make_chunks(pages: list[dict], chunk_size: int = 1000) -> list[dict]:
    chunks = []

    for page in pages:
        text = page["text"]

        for start in range(0, len(text), chunk_size):
            part = text[start : start + chunk_size].strip()
            if part:
                chunks.append({**page, "text": part})

    return chunks


def search_chunks(
    query: str,
    chunks: list[dict],
    top_k: int = 3,
) -> list[dict]:
    if not query.strip() or not chunks:
        return []

    texts = [chunk["text"] for chunk in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")

    try:
        matrix = vectorizer.fit_transform(texts + [query])
    except ValueError:
        return []

    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    indices = scores.argsort()[::-1][:top_k]

    return [
        {**chunks[index], "score": float(scores[index])}
        for index in indices
        if scores[index] > 0
    ]


def rows_to_text(tables: list[dict]) -> str:
    table_text = []

    for table in tables:
        table_text.append(
            "\n".join(
                "\t".join(str(cell or "") for cell in row)
                for row in table["rows"]
            )
        )

    return "\n\n".join(table_text)


st.set_page_config(
    page_title="Multimodal RAG",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Multimodal RAG")
st.caption("Search PDF text, inspect tables and images, and ask questions locally.")

uploaded_pdf = st.file_uploader("Choose a PDF", type=["pdf"])

if uploaded_pdf:
    pdf_bytes = uploaded_pdf.getvalue()

    try:
        pages = extract_text_pages(pdf_bytes, uploaded_pdf.name)
        chunks = make_chunks(pages)
        tables = extract_tables(pdf_bytes)
        reader = PdfReader(BytesIO(pdf_bytes))
        document = pdfium.PdfDocument(pdf_bytes)

        text_tab, table_tab, preview_tab = st.tabs(
            ["Text search", "Tables", "Page previews and Q&A"]
        )

        with text_tab:
            query = st.text_input("Search the PDF text")

            if query:
                results = search_chunks(query, chunks)

                if not results:
                    st.warning("No matching text found. Try different words.")

                for result in results:
                    st.markdown(
                        f"**{result['source']} — page {result['page']}** "
                        f"(match: {result['score']:.2f})"
                    )
                    st.write(result["text"])
                    st.divider()

        with table_tab:
            st.write(f"Tables detected: **{len(tables)}**")

            if not tables:
                st.info(
                    "No tables were detected. Scanned pages may need OCR, "
                    "and some PDF table layouts cannot be detected automatically."
                )

            for table in tables:
                st.markdown(
                    f"**Page {table['page']} · Table {table['table_number']}**"
                )
                st.dataframe(table["rows"], use_container_width=True)

        with preview_tab:
            selected = st.number_input(
                "Page to preview",
                min_value=1,
                max_value=max(1, len(reader.pages)),
                value=1,
                step=1,
            )

            page_number = int(selected)
            page_index = page_number - 1

            # Render the complete page, including vector charts and diagrams.
            page_image = document[page_index].render(scale=1.5).to_pil()
            st.image(
                page_image,
                caption=f"Full page {page_number}",
                use_container_width=True,
            )

            # Display separate image objects embedded in this PDF page.
            st.subheader("Images extracted from this page")

            try:
                embedded_images = list(reader.pages[page_index].images)

                if embedded_images:
                    st.write(f"Images found: **{len(embedded_images)}**")

                    for image_index, image_file in enumerate(
                        embedded_images,
                        start=1,
                    ):
                        st.image(
                            image_file.image,
                            caption=f"Image {image_index} — page {page_number}",
                        )
                else:
                    st.info(
                        "No separate embedded images found. Charts drawn as "
                        "PDF lines and text remain visible in the full-page image."
                    )
            except Exception as image_error:
                st.warning(
                    "Could not extract embedded images from this page: "
                    f"{image_error}"
                )

            st.subheader("Ask about this page")
            question = st.text_input(
                "Question",
                key="page_question",
            )

            if st.button("Get local answer") and question.strip():
                # Search only text from the selected page.
                page_chunks = [
                    chunk for chunk in chunks
                    if chunk["page"] == page_number
                ]
                matches = search_chunks(question, page_chunks, top_k=2)

                text_context = "\n\n".join(
                    item["text"][:700]
                    for item in matches
                )

                # Include tables detected on the selected page.
                page_tables = [
                    table for table in tables
                    if table["page"] == page_number
                ]
                table_context = rows_to_text(page_tables)[:2500]

                # Prepare the page image for Moondream.
                ollama_page = page_image.convert("RGB")
                ollama_page.thumbnail((1400, 1400))

                image_buffer = BytesIO()
                ollama_page.save(
                    image_buffer,
                    format="JPEG",
                    quality=75,
                )
                image_base64 = base64.b64encode(
                    image_buffer.getvalue()
                ).decode("utf-8")

                try:
                    # Stage 1: Moondream describes relevant visual evidence.
                    vision_response = requests.post(
                        "http://localhost:11434/api/chat",
                        json={
                            "model": "moondream",
                            "stream": False,
                            "keep_alive": 0,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": (
                                        "Describe the visible chart, table, or "
                                        "diagram details relevant to this "
                                        f"question: {question}"
                                    ),
                                    "images": [image_base64],
                                }
                            ],
                            "options": {
                                "temperature": 0,
                                "num_predict": 128,
                            },
                        },
                        timeout=300,
                    )
                    vision_response.raise_for_status()

                    vision_payload = vision_response.json()
                    visual_summary = (
                        vision_payload.get("message", {})
                        .get("content", "")
                        .strip()
                    )

                    if not visual_summary:
                        visual_summary = "No visual description was returned."

                    # Stage 2: Qwen answers from the combined evidence.
                    answer_prompt = f"""
Answer the question directly in 1–2 sentences using only the evidence below.
If the evidence does not answer the question, say so. Do not invent details.

Question:
{question}

Text extracted from page {page_number}:
{text_context or "No matching text was retrieved from this page."}

Tables extracted from page {page_number}:
{table_context or "No table was extracted from this page."}

Visual description of page {page_number}:
{visual_summary}

Cite page {page_number} in your answer.
"""

                    answer_response = requests.post(
                        "http://localhost:11434/api/chat",
                        json={
                            "model": "qwen3:0.6b",
                            "stream": False,
                            "think": False,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": answer_prompt,
                                }
                            ],
                            "options": {
                                "temperature": 0,
                                "num_predict": 256,
                            },
                        },
                        timeout=300,
                    )
                    answer_response.raise_for_status()

                    answer_payload = answer_response.json()
                    answer = (
                        answer_payload.get("message", {})
                        .get("content", "")
                        .strip()
                    )

                    if answer:
                        st.markdown("**Answer**")
                        st.write(answer)

                        with st.expander("Evidence used"):
                            st.write("Visual description:")
                            st.write(visual_summary)
                            st.write("Retrieved page text:")
                            st.write(text_context or "No matching text.")
                            st.write("Extracted table:")
                            st.write(table_context or "No table extracted.")
                    else:
                        st.warning(
                            "Qwen returned no final answer. "
                            "Ollama response details:"
                        )
                        st.json(answer_payload)

                except requests.RequestException as request_error:
                    st.error(
                        "Could not reach a local Ollama model. Make sure "
                        "Ollama is running and both models are available. "
                        f"Details: {request_error}"
                    )

    except Exception as error:
        st.error(f"Could not process this PDF: {error}")

else:
    st.info("Upload a PDF to inspect its text, tables, and page visuals.")