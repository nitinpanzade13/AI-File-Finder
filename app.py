import os
import json
import streamlit as st
import clip
import torch
import faiss
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer, InputExample, losses
import PyPDF2
from pdf2image import convert_from_path  # make sure poppler is installed
from torch.utils.data import Dataset, DataLoader
from torch import optim
from tqdm import tqdm
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
import pillow_heif

pillow_heif.register_heif_opener()

# --- Initial Setup ---
st.set_page_config(page_title="AI File Finder", page_icon="✨", layout="wide")
st.title("✨ AI File Finder")
st.markdown("Search through your local images and PDFs using natural language.")

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model Loading (Cached) ---

@st.cache_resource
def load_base_models():
    """Loads the base models that are used across the app."""
    text_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    return text_model, clip_model, preprocess


@st.cache_resource
def load_captioning_model():
    """Loads and caches the BLIP image captioning model and processor."""
    st.info("Loading image captioning model for the first time... This might take a moment.")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base", use_safetensors=True
    ).to(device)
    return processor, model


@st.cache_resource
def load_query_generator_model():
    """
    Loads google/flan-t5-base to generate search queries from PDF chunks.
    """
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    return tokenizer, model


text_model, clip_model_base, clip_preprocess = load_base_models()

# =========================================================
# === Re-Ranker model (Option 2) ==========================
# =========================================================


class ReRanker(torch.nn.Module):
    """
    Small MLP that takes text + image embeddings and outputs a relevance score.
    Input: (text_emb, img_emb) both of shape (N, D)
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(embed_dim * 4, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1),  # raw score (logit)
        )

    def forward(self, text_emb: torch.Tensor, img_emb: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(text_emb - img_emb)
        prod = text_emb * img_emb
        x = torch.cat([text_emb, img_emb, diff, prod], dim=-1)
        logits = self.net(x).squeeze(-1)  # (N,)
        return logits


def get_reranker_path(folder_path: str) -> str:
    return os.path.join(folder_path, "image_reranker.pt")


def feedback_file_path(folder_path: str) -> str:
    return os.path.join(folder_path, "image_feedback.json")


def load_feedback(folder_path: str):
    """
    Loads feedback from session_state first,
    falls back to disk if needed.
    Also resets cache if folder changes.
    """
    # Reset cache if folder changed
    if st.session_state.get("feedback_folder") != folder_path:
        st.session_state["feedback_folder"] = folder_path
        st.session_state["feedback_cache"] = None

    # 1) If already in session_state
    if st.session_state.get("feedback_cache") is not None:
        return st.session_state["feedback_cache"]

    # 2) Load from disk
    path = feedback_file_path(folder_path)
    data = []
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as e:
            st.error(f"Error reading feedback file: {e}")
            data = []

    st.session_state["feedback_cache"] = data
    return data


def log_feedback(folder_path: str, query: str, image_path: str, label: int):
    """
    label: 1 = relevant, 0 = not relevant
    For a given (query, image_path) pair, keep only one entry.
    If it already exists, update label instead of creating duplicates.
    """
    feedback_list = load_feedback(folder_path)  # ensures feedback_cache exists

    # Remove any existing entries for the same (query, image_path)
    new_list = []
    for item in feedback_list:
        if not (item.get("query") == query and item.get("image_path") == image_path):
            new_list.append(item)

    # Add the updated / new entry
    new_list.append({"query": query, "image_path": image_path, "label": int(label)})

    # Update cache + disk
    st.session_state["feedback_cache"] = new_list
    try:
        path = feedback_file_path(folder_path)
        with open(path, "w") as f:
            json.dump(new_list, f, indent=2)
    except Exception as e:
        st.error(f"Error saving feedback file: {e}")


class FeedbackDataset(Dataset):
    """
    Dataset for re-ranking training using user feedback.
    Each item: (text_tokens, image_tensor, label)
    """

    def __init__(self, feedback_list, preprocess):
        self.feedback_list = feedback_list
        self.preprocess = preprocess

    def __len__(self):
        return len(self.feedback_list)

    def __getitem__(self, idx):
        item = self.feedback_list[idx]
        query = item["query"]
        image_path = item["image_path"]
        label = float(item["label"])

        text_tokens = clip.tokenize([query])[0]  # (77,)
        image = self.preprocess(Image.open(image_path).convert("RGB"))
        return text_tokens, image, label


def train_reranker(folder_path: str, clip_model, preprocess, epochs: int = 5, batch_size: int = 16):
    feedback = load_feedback(folder_path)
    if not feedback:
        st.warning("No feedback found yet. Mark some results as 👍 / 👎 first.")
        return

    n_pos = sum(1 for x in feedback if x["label"] == 1)
    n_neg = sum(1 for x in feedback if x["label"] == 0)
    if n_pos == 0 or n_neg == 0:
        st.warning("Need at least one positive and one negative feedback example.")
        return

    dataset = FeedbackDataset(feedback, preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    embed_dim = clip_model.text_projection.shape[1]  # 512 for ViT-B/32
    reranker = ReRanker(embed_dim).to(device)

    optimizer = optim.AdamW(reranker.parameters(), lr=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    total_steps = epochs * len(dataloader)
    step = 0
    progress_bar = st.progress(0, text="Training re-ranker...")

    clip_model.eval()
    reranker.train()

    for epoch in range(epochs):
        for text_tokens, images, labels in dataloader:
            text_tokens = text_tokens.to(device)
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                text_emb = clip_model.encode_text(text_tokens)
                img_emb = clip_model.encode_image(images)
                text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

            logits = reranker(text_emb, img_emb)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            progress_bar.progress(
                min(step / total_steps, 1.0),
                text=f"Training re-ranker... Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}",
            )

    progress_bar.empty()
    torch.save(reranker.state_dict(), get_reranker_path(folder_path))
    st.success("Re-ranker training complete! It will now re-order your image results (if enabled).")

# --- BLIP Captioning + CLIP Fine-tuning (existing personalization) ---


class CaptionGenerator:
    """Generates and caches captions for images in a folder using batch processing."""

    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.caption_file = os.path.join(folder_path, ".generated_captions.json")
        self.captions = self.load_captions()
        self.processor, self.model = load_captioning_model()

    def load_captions(self):
        if os.path.exists(self.caption_file):
            with open(self.caption_file, "r") as f:
                return json.load(f)
        return {}

    def save_captions(self):
        with open(self.caption_file, "w") as f:
            json.dump(self.captions, f, indent=2)

    def generate_captions_for_batch(self, image_filenames):
        """Generates captions for a batch of images."""
        raw_images = []
        valid_filenames = []
        for filename in image_filenames:
            try:
                img_path = os.path.join(self.folder_path, filename)
                image = Image.open(img_path).convert("RGB").resize((224, 224))
                raw_images.append(image)
                valid_filenames.append(filename)
            except Exception:
                self.captions[filename] = "could not read image"

        if not raw_images:
            return

        try:
            inputs = self.processor(raw_images, return_tensors="pt").to(device)
            out = self.model.generate(**inputs, max_new_tokens=50)
            captions = self.processor.batch_decode(out, skip_special_tokens=True)

            for filename, caption in zip(valid_filenames, captions):
                self.captions[filename] = caption
        except Exception as e:
            st.error(f"Error during batch captioning: {e}")


class ImageTextDataset(Dataset):
    """Dataset for fine-tuning using AI-generated captions with batch generation."""

    def __init__(self, folder_path, preprocess, caption_generator):
        self.folder_path = folder_path
        self.preprocess = preprocess
        self.image_paths = [
            f
            for f in os.listdir(folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ]
        self.caption_generator = caption_generator

        new_images = [f for f in self.image_paths if f not in self.caption_generator.captions]

        if new_images:
            st.info(f"Stage 1: Found {len(new_images)} new images. Generating captions...")
            batch_size = 16

            progress_bar = st.progress(0, text="Analyzing images in batches...")
            for i in tqdm(range(0, len(new_images), batch_size), desc="Generating Captions in Batches"):
                batch_filenames = new_images[i : i + batch_size]
                self.caption_generator.generate_captions_for_batch(batch_filenames)
                progress_bar.progress(
                    min((i + batch_size) / len(new_images), 1.0),
                    text=f"Analyzed {min(i + batch_size, len(new_images))}/{len(new_images)} images",
                )

            self.caption_generator.save_captions()
            progress_bar.empty()
            st.success("Caption generation complete!")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_filename = self.image_paths[idx]
        img_path = os.path.join(self.folder_path, img_filename)
        image = self.preprocess(Image.open(img_path).convert("RGB"))
        text = self.caption_generator.captions.get(img_filename, "a photo")
        return image, text


def fine_tune_model(model, preprocess, folder_path, epochs=1):
    caption_generator = CaptionGenerator(folder_path)
    dataset = ImageTextDataset(folder_path, preprocess, caption_generator)
    if len(dataset) == 0:
        st.error("No images found to fine-tune on.")
        return

    st.info(f"Stage 2: Starting fine-tuning for {epochs} epoch(s)...")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=5e-7)
    loss_fn = torch.nn.CrossEntropyLoss()

    total_steps = epochs * len(dataloader)
    step = 0
    progress_bar = st.progress(0, text="Fine-tuning model...")

    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for images, texts in pbar:
            images = images.to(device)
            text_tokens = clip.tokenize(list(texts)).to(device)

            logits_per_image, logits_per_text = model(images, text_tokens)
            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
            total_loss = (loss_fn(logits_per_image, ground_truth) + loss_fn(logits_per_text, ground_truth)) / 2

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            step += 1
            pbar.set_postfix({"loss": total_loss.item()})
            progress_bar.progress(
                min(step / total_steps, 1.0),
                text=f"Epoch {epoch+1}, Step {step}/{total_steps}",
            )

    progress_bar.empty()
    st.success("Fine-tuning complete!")

# --- Per-folder PDF model helpers (FLAN-T5 + MPNet) ---


def get_pdf_model_path(folder_path: str) -> str:
    """Directory where the fine-tuned PDF model for this folder will be stored."""
    return os.path.join(folder_path, "pdf_model")


def extract_pdf_page_chunks(folder_path: str, min_chars: int = 100):
    """
    Extracts text per page for each PDF in the folder.
    Returns a list of dicts: {pdf_path, pdf_filename, page, text}.
    Skips pages with very little or no text.
    """
    chunks = []
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]

    for fname in pdf_files:
        pdf_path = os.path.join(folder_path, fname)
        try:
            reader = PyPDF2.PdfReader(pdf_path)
        except Exception as e:
            st.warning(f"Could not read PDF {fname}: {e}")
            continue

        for page_idx, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
                text = text.strip()
            except Exception:
                text = ""

            if len(text) < min_chars:
                # Skip empty or too short pages
                continue

            chunks.append(
                {
                    "pdf_path": pdf_path,
                    "pdf_filename": fname,
                    "page": page_idx,  # 0-based
                    "text": text,
                }
            )

    return chunks


def generate_queries_for_chunk(chunk_text: str, num_queries: int = 3):
    """
    Uses google/flan-t5-base to generate a few possible search queries
    a user might type to find this chunk.
    """
    tokenizer, model = load_query_generator_model()

    prompt = (
        "Read the following study material or document content and generate "
        f"{num_queries} different short search queries a user might type to find it.\n\n"
        f"CONTENT:\n{chunk_text}\n\nQUERIES:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            num_beams=4,
            do_sample=False,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Split by lines, basic cleaning
    lines = [l.strip(" -•\t") for l in text.split("\n") if l.strip()]
    uniq = []
    for l in lines:
        if l and l not in uniq:
            uniq.append(l)
    return uniq[:num_queries]


def build_pdf_auto_training_data(folder_path: str, num_queries_per_chunk: int = 3, max_chunks: int = 100):
    """
    For the given folder, extract page-level chunks from PDFs,
    auto-generate queries for each chunk using FLAN-T5,
    and store (query, chunk_text, pdf_filename, page) pairs in a JSON file.
    """
    output_file = os.path.join(folder_path, "pdf_auto_training_data.json")

    st.info("Extracting PDF page chunks...")
    chunks = extract_pdf_page_chunks(folder_path)
    if not chunks:
        st.error("No suitable PDF text chunks found in this folder.")
        return

    # Optionally limit total chunks to control time
    if len(chunks) > max_chunks:
        st.warning(f"Too many chunks ({len(chunks)}). Limiting to first {max_chunks} for auto-labeling.")
        chunks = chunks[:max_chunks]

    training_pairs = []

    progress_bar = st.progress(0, text="Generating queries for chunks...")
    total = len(chunks)

    for i, ch in enumerate(chunks):
        chunk_text = ch["text"]
        pdf_filename = ch["pdf_filename"]
        page_idx = ch["page"]

        # Generate queries for this chunk
        queries = generate_queries_for_chunk(chunk_text, num_queries=num_queries_per_chunk)

        for q in queries:
            training_pairs.append(
                {
                    "query": q,
                    "pdf_filename": pdf_filename,
                    "page": page_idx,
                    "chunk_text": chunk_text,
                }
            )

        progress_bar.progress((i + 1) / total, text=f"Processed {i+1}/{total} chunks...")

    progress_bar.empty()

    if not training_pairs:
        st.error("No training pairs were generated. Try adjusting min_chars or num_queries_per_chunk.")
        return

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(training_pairs, f, indent=2, ensure_ascii=False)

    st.success(f"Auto training data saved to: {output_file}")
    st.info(f"Total training pairs: {len(training_pairs)}")


def train_pdf_model_from_auto_data(
    folder_path: str,
    base_model_name: str = "sentence-transformers/all-mpnet-base-v2",
    epochs: int = 2,
    batch_size: int = 4,
):
    """
    Fine-tunes a SentenceTransformer model on auto-generated (query, chunk_text) pairs
    for THIS folder, and saves it under folder_path/pdf_model.
    """
    data_file = os.path.join(folder_path, "pdf_auto_training_data.json")
    if not os.path.exists(data_file):
        st.error(f"Auto training data not found: {data_file}")
        return

    with open(data_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    train_examples = []
    for entry in raw:
        query = entry.get("query")
        chunk_text = entry.get("chunk_text")
        if not query or not chunk_text:
            continue
        train_examples.append(InputExample(texts=[query, chunk_text]))

    if not train_examples:
        st.error("No valid training examples found in pdf_auto_training_data.json.")
        return

    st.info(f"Training on {len(train_examples)} (query, chunk) pairs...")

    model = SentenceTransformer(base_model_name, device=device)

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = max(1, int(0.1 * len(train_dataloader) * epochs))

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
    )

    save_path = get_pdf_model_path(folder_path)
    os.makedirs(save_path, exist_ok=True)
    model.save(save_path)
    st.success(f"Fine-tuned folder PDF model saved to: {save_path}")

# --- FAISS Indexing ---


def build_faiss_index_images(folder_path, model, preprocess, is_personalized):
    index_name = "image_index_personalized.faiss" if is_personalized else "image_index.faiss"
    index_file = os.path.join(folder_path, index_name)
    paths_file = os.path.join(folder_path, "img_paths.json")

    if os.path.exists(index_file) and os.path.exists(paths_file):
        faiss_index = faiss.read_index(index_file)
        with open(paths_file, "r") as f:
            img_paths = json.load(f)
        return faiss_index, img_paths

    embeddings, paths = [], []
    with st.spinner(f"🔧 Creating new {'personalized' if is_personalized else 'default'} image index..."):
        all_files = [
            f
            for f in os.listdir(folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ]
        for f in tqdm(all_files, desc="Indexing Images"):
            path = os.path.join(folder_path, f)
            try:
                image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model.encode_image(image).cpu().numpy().astype("float32")
                embeddings.append(emb)
                paths.append(path)
            except Exception:
                continue

    if not embeddings:
        return None, []

    embeddings = np.vstack(embeddings)
    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)
    faiss_index.add(embeddings)
    faiss.write_index(faiss_index, index_file)
    with open(paths_file, "w") as f:
        json.dump(paths, f)

    return faiss_index, paths


def build_faiss_index_pdfs(folder_path):
    """
    Old whole-document PDF index (kept for compatibility, not used in page-level search now).
    """
    index_file = os.path.join(folder_path, "pdf_index.faiss")
    paths_file = os.path.join(folder_path, "pdf_paths.json")

    if os.path.exists(index_file) and os.path.exists(paths_file):
        faiss_index = faiss.read_index(index_file)
        with open(paths_file, "r") as f:
            pdf_paths = json.load(f)
        return faiss_index, pdf_paths

    embeddings, paths = [], []
    with st.spinner("🔧 Creating new PDF index..."):
        for f in tqdm(os.listdir(folder_path), desc="Indexing PDFs"):
            if f.lower().endswith(".pdf"):
                path = os.path.join(folder_path, f)
                try:
                    pdf_reader = PyPDF2.PdfReader(path)
                    text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
                    if text.strip():
                        emb = text_model.encode(text, convert_to_numpy=True).astype("float32")
                        embeddings.append(emb)
                        paths.append(path)
                except Exception:
                    continue

    if not embeddings:
        return None, []

    embeddings = np.vstack(embeddings)
    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)
    faiss_index.add(embeddings)
    faiss.write_index(faiss_index, index_file)
    with open(paths_file, "w") as f:
        json.dump(paths, f)

    return faiss_index, paths


def build_pdf_image_index(folder_path, clip_model, preprocess):
    """
    Builds/loads a FAISS index over PDF pages treated as images.
    Each entry corresponds to a (pdf_path, page_number).
    """
    index_file = os.path.join(folder_path, "pdf_page_index.faiss")
    meta_file = os.path.join(folder_path, "pdf_page_meta.json")

    # If already exists, load and return
    if os.path.exists(index_file) and os.path.exists(meta_file):
        faiss_index = faiss.read_index(index_file)
        with open(meta_file, "r", encoding="utf-8") as f:
            page_meta = json.load(f)
        return faiss_index, page_meta

    embeddings = []
    page_meta = []

    with st.spinner("🔧 Creating new PDF page (image) index..."):
        pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
        for fname in tqdm(pdf_files, desc="Indexing PDF pages as images"):
            pdf_path = os.path.join(folder_path, fname)
            try:
                pages = convert_from_path(pdf_path, dpi=120)
            except Exception as e:
                st.warning(f"Could not render PDF {fname}: {e}")
                continue

            for page_num, pil_img in enumerate(pages):
                try:
                    img_tensor = preprocess(pil_img.convert("RGB")).unsqueeze(0).to(device)
                    with torch.no_grad():
                        emb = clip_model.encode_image(img_tensor)
                        emb = emb / emb.norm(dim=-1, keepdim=True)
                        emb = emb.cpu().numpy().astype("float32")
                    embeddings.append(emb)
                    page_meta.append(
                        {
                            "pdf_path": pdf_path,
                            "page": page_num,  # 0-based index
                        }
                    )
                except Exception:
                    continue

    if not embeddings:
        return None, []

    embeddings = np.vstack(embeddings)
    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)
    faiss_index.add(embeddings)

    faiss.write_index(faiss_index, index_file)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(page_meta, f, indent=2)

    return faiss_index, page_meta


def build_faiss_index_pdf_chunks(folder_path):
    """
    Build/load a FAISS index over page-level text chunks of PDFs in this folder.
    Uses fine-tuned folder PDF model if available, else falls back to global text_model.
    Stores metadata as {pdf_path, page}.
    """
    index_file = os.path.join(folder_path, "pdf_chunk_index.faiss")
    meta_file = os.path.join(folder_path, "pdf_chunk_meta.json")

    pdf_model_path = get_pdf_model_path(folder_path)
    if os.path.exists(pdf_model_path):
        pdf_text_model = SentenceTransformer(pdf_model_path, device=device)
    else:
        pdf_text_model = text_model

    if os.path.exists(index_file) and os.path.exists(meta_file):
        faiss_index = faiss.read_index(index_file)
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return faiss_index, meta

    chunks = extract_pdf_page_chunks(folder_path)
    if not chunks:
        return None, []

    embeddings = []
    meta = []

    with st.spinner("🔧 Creating new PDF chunk (page) index..."):
        for ch in tqdm(chunks, desc="Indexing PDF text chunks"):
            text_chunk = ch["text"]
            pdf_path = ch["pdf_path"]
            page_idx = ch["page"]

            try:
                emb = pdf_text_model.encode(text_chunk, convert_to_numpy=True).astype("float32")
                embeddings.append(emb)
                meta.append({"pdf_path": pdf_path, "page": page_idx})
            except Exception:
                continue

    if not embeddings:
        return None, []

    embeddings = np.vstack(embeddings)
    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)
    faiss_index.add(embeddings)

    faiss.write_index(faiss_index, index_file)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return faiss_index, meta

# --- Streamlit UI ---


st.sidebar.header("📁 Configuration")
folder_path = st.sidebar.text_input("Enter Folder Path to Search", key="folder_path")

model_choice = "Default"
if folder_path and os.path.exists(folder_path):
    personalized_model_path = os.path.join(folder_path, "personalized_model.pt")
    if os.path.exists(personalized_model_path):
        model_choice = st.sidebar.radio(
            "Choose Image Model", ("Default", "Personalized"), key="model_choice"
        )

use_reranker = False
if folder_path and os.path.exists(folder_path):
    if os.path.exists(get_reranker_path(folder_path)):
        use_reranker = st.sidebar.checkbox(
            "Use learned image re-ranking (if available)", value=True, key="use_reranker"
        )

mode = st.radio(
    "Select content to search:", ["🖼 Images", "📄 PDF Documents"], horizontal=True
)
st.divider()

# ============= IMAGE MODE =============
if mode == "🖼 Images":
    if not folder_path or not os.path.exists(folder_path):
        st.warning("Please enter a valid folder path in the sidebar to begin.")
    else:
        # --- Optional CLIP personalization ---
        with st.expander("✨ Personalize Image Search Model (Optional)"):
            st.markdown(
                """
            Fine-tune the model on the images in this folder to improve search relevance for your specific content.
            This involves two stages:
            1.  Caption Generation: An AI model analyzes each image and writes a description.
            2.  Fine-Tuning: The CLIP model is trained on your images and their new descriptions.
            
            Note: This is a one-time process per folder and can be computationally intensive.
            """
            )
            if st.button("Start Personalization Process"):
                with st.spinner("Loading base model for tuning..."):
                    model_to_tune, preprocess = clip.load("ViT-B/32", device=device)
                fine_tune_model(model_to_tune, preprocess, folder_path)
                torch.save(model_to_tune.state_dict(), personalized_model_path)
                st.success(
                    "Personalized model saved! Please refresh the page and select 'Personalized' in the sidebar."
                )

        # --- Train Re-Ranker (feedback-based) ---
        with st.expander("🔁 Train Re-Ranker from Feedback (NEW)"):
            st.markdown(
                """
            This trains a small model that learns from your 👍 / 👎 labels on search results.
            Over time, it will reorder results to better match your preferences.
            """
            )

            if folder_path and os.path.exists(folder_path):
                fb = load_feedback(folder_path)
                n_total = len(fb)
                n_pos = sum(1 for x in fb if x["label"] == 1)
                n_neg = sum(1 for x in fb if x["label"] == 0)
                st.info(f"Feedback collected: {n_total} samples ({n_pos} 👍, {n_neg} 👎)")

            if st.button("Train Re-Ranker Now"):
                with st.spinner("Training re-ranker model..."):
                    train_reranker(folder_path, clip_model_base, clip_preprocess)

        st.header("🖼 Image Search")
        query = st.text_input("Search for images...", key="img_query")
        top_k_images = st.slider(
            "Number of image results to show:", 1, 5, 5, key="img_slider"
        )

        # --- Run search when button pressed: store results in session_state ---
        if st.button("Search Images", key="img_search_btn"):
            if not query.strip():
                st.warning("Please enter a query.")
            else:
                is_personalized = model_choice == "Personalized"
                model_to_use = clip_model_base
                if is_personalized:
                    try:
                        personalized_model_path = os.path.join(
                            folder_path, "personalized_model.pt"
                        )
                        model_to_use.load_state_dict(
                            torch.load(personalized_model_path, map_location=device)
                        )
                        st.sidebar.success("✅ Personalized model loaded!")
                    except Exception as e:
                        st.sidebar.error(f"Could not load personalized model: {e}")
                        st.stop()
                else:
                    st.sidebar.success("✅ Default model loaded!")

                index, img_paths = build_faiss_index_images(
                    folder_path, model_to_use, clip_preprocess, is_personalized
                )
                if index is None:
                    st.error("⚠ No images found or indexed.")
                    st.stop()

                text_tokens = clip.tokenize([query]).to(device)
                with torch.no_grad():
                    text_features = model_to_use.encode_text(text_tokens)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                text_emb = text_features.cpu().numpy().astype("float32")
                faiss.normalize_L2(text_emb)

                D, I = index.search(text_emb, top_k_images)

                # --- Apply re-ranker if enabled and available ---
                if use_reranker and os.path.exists(get_reranker_path(folder_path)):
                    reranker = ReRanker(embed_dim=text_features.shape[1]).to(device)
                    reranker.load_state_dict(
                        torch.load(get_reranker_path(folder_path), map_location=device)
                    )
                    reranker.eval()

                    img_emb_list = []
                    with torch.no_grad():
                        for idx in I[0]:
                            img_path_i = img_paths[idx]
                            image = clip_preprocess(
                                Image.open(img_path_i).convert("RGB")
                            ).unsqueeze(0).to(device)
                            img_feat = model_to_use.encode_image(image)
                            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                            img_emb_list.append(img_feat)

                        img_embs = torch.cat(img_emb_list, dim=0)  # (K, D)
                        text_rep = text_features.repeat(img_embs.size(0), 1)
                        rerank_scores = reranker(text_rep, img_embs).cpu().numpy()  # (K,)

                    order = np.argsort(-rerank_scores)
                    I[0] = I[0][order]
                    D[0] = rerank_scores[order]

                # Save results to session_state so they persist after rerun
                results = []
                for idx, score in zip(I[0], D[0]):
                    results.append({"img_path": img_paths[idx], "score": float(score)})

                st.session_state["last_image_results"] = {
                    "query": query,
                    "results": results,
                    "folder_path": folder_path,
                }

        # --- Always display last results (if any) and feedback buttons ---
        if "last_image_results" in st.session_state:
            data = st.session_state["last_image_results"]

            # If folder changed, discard old results
            if data.get("folder_path") != folder_path:
                st.session_state.pop("last_image_results")
            else:
                current_query = data["query"]
                st.subheader(f"🔎 Results for: {current_query}")
                results = data["results"]

                # 🔹 Build a quick lookup for feedback: (query, image_path) -> label
                fb = load_feedback(folder_path)
                fb_map = {(item["query"], item["image_path"]): item["label"] for item in fb}

                # Filter out images that are explicitly labeled as NOT relevant for THIS query
                filtered_results = []
                for r in results:
                    img_path_i = r["img_path"]
                    label = fb_map.get((current_query, img_path_i))
                    if label == 0:  # 0 = Not Relevant for this query
                        continue  # skip this image
                    filtered_results.append(r)

                if not filtered_results:
                    st.info(
                        "No results to display (some may be hidden as 'Not Relevant' for this query)."
                    )
                else:
                    cols = st.columns(5)
                    for i, item in enumerate(filtered_results):
                        img_path_i = item["img_path"]
                        score = item["score"]
                        col = cols[i % 5]
                        col.image(img_path_i, caption=f"Score: {score:.2f}")

                        col.write("Feedback:")
                        btn_suffix = f"{i}_{os.path.basename(img_path_i)}"

                        if col.button("👍 Relevant", key=f"rel_{btn_suffix}"):
                            log_feedback(folder_path, current_query, img_path_i, label=1)
                            st.success(f"Marked as relevant: {os.path.basename(img_path_i)}")

                        if col.button("👎 Not Relevant", key=f"irr_{btn_suffix}"):
                            log_feedback(folder_path, current_query, img_path_i, label=0)
                            st.info(f"Marked as not relevant: {os.path.basename(img_path_i)}")

# ============= PDF MODE =============
elif mode == "📄 PDF Documents":
    if not folder_path or not os.path.exists(folder_path):
        st.warning("Please enter a valid folder path in the sidebar to begin.")
    else:
        st.header("📄 PDF Document Search (Personalized per folder)")

        # ----- Auto-training section -----
        with st.expander("🧠 Auto-train folder PDF model with FLAN-T5 + MPNet"):
            st.markdown(
                """
            This will:
            1. Split each PDF in this folder into page-level text chunks.
            2. Use **google/flan-t5-base** to auto-generate search queries for each chunk.
            3. Fine-tune a **folder-specific MPNet model** (`pdf_model/`) on those (query, chunk) pairs.
            
            After training, PDF search in this folder becomes personalized and can return **exact PDF + page**.
            """
            )

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("① Generate training data", key="btn_gen_pdf_auto_data"):
                    build_pdf_auto_training_data(
                        folder_path, num_queries_per_chunk=3, max_chunks=80
                    )
            with col_b:
                if st.button("② Train folder PDF model", key="btn_train_pdf_model"):
                    train_pdf_model_from_auto_data(folder_path)

        # --- Hybrid PDF search controls ---
        pdf_query = st.text_input(
            "Search through PDFs (page-level text + page images)...", key="pdf_query"
        )

        col1, col2 = st.columns(2)
        with col1:
            top_k_text = st.slider(
                "Number of TEXT-based page results:", 1, 20, 10, key="pdf_text_slider"
            )
        with col2:
            top_k_pages = st.slider(
                "Number of IMAGE-based page results:", 1, 10, 5, key="pdf_img_slider"
            )

        if st.button("Search PDFs", key="pdf_search_btn"):
            if not pdf_query.strip():
                st.warning("Please enter a query.")
            else:
                # --- TEXT-BASED PAGE-LEVEL PDF SEARCH ---
                pdf_chunk_index, pdf_chunk_meta = build_faiss_index_pdf_chunks(folder_path)
                if pdf_chunk_index is None:
                    st.error("⚠ No PDF text chunks found or indexed.")
                else:
                    pdf_model_path = get_pdf_model_path(folder_path)
                    if os.path.exists(pdf_model_path):
                        pdf_text_model = SentenceTransformer(pdf_model_path, device=device)
                        st.info("Using personalized PDF model for this folder.")
                    else:
                        pdf_text_model = text_model
                        st.info(
                            "Using base all-mpnet model (no folder-specific fine-tuning found)."
                        )

                    query_emb = (
                        pdf_text_model.encode(
                            pdf_query, convert_to_numpy=True
                        ).astype("float32")
                    ).reshape(1, -1)
                    faiss.normalize_L2(query_emb)
                    D_text, I_text = pdf_chunk_index.search(query_emb, top_k_text)

                # --- IMAGE-BASED PAGE SEARCH (via CLIP) ---
                pdf_page_index, pdf_page_meta = build_pdf_image_index(
                    folder_path, clip_model_base, clip_preprocess
                )
                use_image_search = pdf_page_index is not None and len(pdf_page_meta) > 0

                if not use_image_search:
                    st.warning(
                        "No PDF page image index available (maybe no PDFs or pdf2image/poppler not configured)."
                    )
                else:
                    text_tokens = clip.tokenize([pdf_query]).to(device)
                    with torch.no_grad():
                        clip_text_feat = clip_model_base.encode_text(text_tokens)
                        clip_text_feat = clip_text_feat / clip_text_feat.norm(
                            dim=-1, keepdim=True
                        )
                    clip_query_emb = clip_text_feat.cpu().numpy().astype("float32")
                    faiss.normalize_L2(clip_query_emb)
                    D_img, I_img = pdf_page_index.search(clip_query_emb, top_k_pages)

                # --- DISPLAY RESULTS ---
                if "D_text" in locals() and pdf_chunk_index is not None:
                    st.subheader("🔎 Top Matches by PDF TEXT (page-level)")
                    for idx, score in zip(I_text[0], D_text[0]):
                        meta = pdf_chunk_meta[idx]
                        pdf_path = meta["pdf_path"]
                        page_idx = meta["page"]
                        st.markdown(
                            f"Score: {score:.4f} — **{os.path.basename(pdf_path)}**, page **{page_idx + 1}**"
                        )

                if use_image_search:
                    st.subheader("🖼 Top Matches by PDF PAGE Images")
                    for idx, score in zip(I_img[0], D_img[0]):
                        meta = pdf_page_meta[idx]
                        pdf_path = meta["pdf_path"]
                        page_num = meta["page"]  # 0-based

                        st.markdown(
                            f"Score: {score:.4f} — **{os.path.basename(pdf_path)}**, page **{page_num + 1}**"
                        )

                        try:
                            page_img = convert_from_path(
                                pdf_path,
                                dpi=100,
                                first_page=page_num + 1,
                                last_page=page_num + 1,
                            )[0]
                            st.image(page_img, width=300)
                        except Exception as e:
                            st.caption(f"(Could not render preview: {e})")