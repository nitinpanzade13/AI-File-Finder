📁 AI File Finder – Intelligent Local File Search using CLIP, BLIP, MPNet & FAISS

AI File Finder is a hybrid PDF + Image search system that lets users instantly search their local files using natural language.
It combines image understanding, document text embedding, PDF page-level visual search, and user-feedback-based reranking to deliver fast and highly accurate search results.

This project supports:

✅ Image Search
✅ PDF Text Search
✅ PDF Page Visual Search
✅ Per-Folder Personalized Models
✅ BLIP Captions for Images
✅ FLAN-T5 Auto Query Generation for PDF Training
✅ FAISS Indexing for Millisecond Retrieval
✅ Feedback-Based Re-Ranking
✅ Clean Streamlit UI

🚀 Features
🔍 1. Image Search (CLIP + BLIP + FAISS)

Captures image meaning using CLIP embeddings

Generates high-quality captions using BLIP

Stores image embeddings in FAISS for instant lookup

Supports user feedback to re-rank results

Allows per-folder fine-tuned CLIP models for personalization

📄 2. PDF Text Search (MPNet + Chunking + FAISS)

Extracts text from multi-page PDFs

Splits documents into page-level & chunk-level embeddings

Uses MPNet (all-mpnet-base-v2) for semantic text search

Supports FLAN-T5 generated training data for automatic fine-tuning

Can locate:

the correct PDF

the exact page containing the answer

🖼️ 3. PDF Page Image Search (CLIP + pdf2image)

Converts each PDF page into an image

Applies CLIP image encoder

Allows visual search, useful for:

certificates

diagrams

handwritten scanned notes

presentations (PPT/PDF)

💡 4. Re-Ranker with Human Feedback

When the user clicks “Relevant” or “Not Relevant”, the system collects feedback

A lightweight neural network learns the user’s preferences

Subsequent searches show sharper, personalized results

🧠 5. Personalized Models Per Folder

Each folder (e.g., Study Materials, Certificates, Projects) gets its own:

Custom BLIP captions

Fine-tuned CLIP for images

Fine-tuned MPNet for PDF text

Unique FAISS indexes

Metadata for fast search

This makes searches more accurate and context-aware.

🖥️ 6. Streamlit Web App

Clean, simple interface

Folder picker

Mode selection (Images / PDFs)

Search bar

Real-time ranked results

Feedback buttons

Re-train buttons for each module

🏗️ System Architecture Summary

The system consists of:

Streamlit UI

Three Search Pipelines

Image pipeline

PDF text pipeline

PDF page image pipeline

FAISS Index Storage

Re-Ranker with Feedback

Folder-specific fine-tuned models

See the full architecture diagram in the repo.

📁 Tech Stack
Category	Tools
UI	Streamlit
Image Models	CLIP, BLIP
Text Models	MPNet, FLAN-T5
Vector Storage	FAISS
PDF Processing	PyPDF2, pdf2image, Poppler
Training	PyTorch
Metadata	JSON
📚 Project Goals

Create a fast, local, privacy-friendly AI search assistant

Enable semantic (meaning-based) file search

Personalize search results using feedback and fine-tuning

Make PDF and image retrieval effortless

🛠️ Installation
pip install -r requirements.txt


Ensure Poppler is installed for PDF-to-image conversion.

▶️ Running the App
streamlit run app.py

🤝 Contributions

Pull requests and improvements are welcome!

⭐ Like this project?

If it helped you, star the repo 🌟 and share it!
