import urllib.parse

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

# Try to use streamlit-cropper if available
try:
    from streamlit_cropper import st_cropper

    HAS_CROPPER = True
except ImportError:
    HAS_CROPPER = False


# ---------- Model loading ----------

@st.cache_resource
def load_trocr_model():
    """
    Load TrOCR processor + model once and cache them.
    Using microsoft/trocr-base-printed (printed text).
    """
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    model.eval()
    return processor, model


# ---------- Image helpers ----------

def read_pil_and_bgr(uploaded_file):
    """Return both PIL (for UI) and OpenCV BGR image."""
    pil_img = Image.open(uploaded_file).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return pil_img, img_bgr


def crop_image_with_box(img_bgr, x1, y1, x2, y2):
    """Crop the image to the given rectangle (clamped to image bounds)."""
    h, w = img_bgr.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(1, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(1, min(h, y2))
    if x2 <= x1:
        x2 = min(w, x1 + 10)
    if y2 <= y1:
        y2 = min(h, y1 + 10)
    return img_bgr[y1:y2, x1:x2]


# ---------- OCR pipeline (TrOCR) ----------

def preprocess_for_trocr(img_bgr):
    """
    Preprocess for TrOCR:
    - convert to RGB
    - mild denoise
    - resize so longest side <= 1024
    - keep as PIL.Image for processor
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.fastNlMeansDenoisingColored(img_rgb, None, 3, 3, 7, 21)

    h, w = img_rgb.shape[:2]
    max_side = max(h, w)
    if max_side > 1024:
        scale = 1024 / max_side
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pil_img = Image.fromarray(img_rgb)
    return pil_img


def trocr_ocr(img_bgr):
    """Run TrOCR on a cropped crossword area and return cleaned text."""
    processor, model = load_trocr_model()
    pil_img = preprocess_for_trocr(img_bgr)
    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values

    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_new_tokens=64)

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    text = " ".join(text.strip().split())
    return text


def segment_text_lines(img_bgr, max_lines=4, min_line_height=8):
    """
    Segment a clue box into up to `max_lines` horizontal text lines.
    Returns a list of (y_start, y_end) row indices.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = 255 - bw

    h, w = inv.shape
    row_counts = np.sum(inv > 0, axis=1)

    threshold = 0.02 * w
    active = row_counts > threshold

    lines = []
    in_line = False
    start = 0

    for y, val in enumerate(active):
        if val and not in_line:
            in_line = True
            start = y
        elif not val and in_line:
            end = y
            if end - start >= min_line_height:
                lines.append((start, end))
            in_line = False

    if in_line:
        end = h
        if end - start >= min_line_height:
            lines.append((start, end))

    if len(lines) > max_lines:
        lines = lines[:max_lines]

    return lines


def ocr_box_auto(img_bgr, max_lines=4):
    """
    OCR for a clue box:
    - 1 line: OCR whole box
    - 2–4 lines: OCR per line, merge, handle '-' as continuation.
    """
    lines = segment_text_lines(img_bgr, max_lines=max_lines)

    if len(lines) <= 1:
        return trocr_ocr(img_bgr).strip()

    line_texts = []
    for (y1, y2) in lines:
        line_img = img_bgr[y1:y2, :]
        txt = trocr_ocr(line_img).strip()
        if txt:
            line_texts.append(txt)

    if not line_texts:
        return ""

    merged = ""
    for s in line_texts:
        s = s.strip()
        if not s:
            continue

        if s.endswith("-"):
            s = s[:-1].rstrip()
            merged += s
        else:
            if merged and not merged.endswith(" "):
                merged += " "
            merged += s

    return " ".join(merged.strip().split())


def detect_clue(ocr_output: str):
    """Clean up a single clue string."""
    line = ocr_output.strip()
    if not line:
        return ""

    cleaned = "".join(
        ch for ch in line
        if ch.isalpha() or ch.isspace() or ch in "-'/0123456789?!.,"
    ).strip()

    if len(cleaned) < 3:
        return ""

    alpha_count = sum(ch.isalpha() for ch in cleaned)
    if alpha_count / max(len(cleaned), 1) < 0.6:
        return ""

    return " ".join(cleaned.split())


# ---------- Google search helper ----------

def google_search_url_for_clue(clue_text, length):
    base = "https://www.google.com/search?q="
    if length and length > 0:
        query = f"korsord {clue_text} {length} bokstäver"
    else:
        query = f"korsord {clue_text}"
    return base + urllib.parse.quote_plus(query)


# ---------- Streamlit app ----------

def main():
    st.set_page_config(page_title="Korsordslösaren", layout="wide")

    # Slightly tighter padding on smaller screens (harmless)
    st.markdown(
        """
        <style>
        @media (max-width: 1200px) {
          .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # State init
    if "saved_queries" not in st.session_state:
        st.session_state["saved_queries"] = []
    if "current_text" not in st.session_state:
        st.session_state["current_text"] = ""
    if "current_len" not in st.session_state:
        st.session_state["current_len"] = 3          # default 3 boxes
    if "image" not in st.session_state:
        st.session_state["image"] = None
    if "image_name" not in st.session_state:
        st.session_state["image_name"] = ""
    if "confirm_reset" not in st.session_state:
        st.session_state["confirm_reset"] = False
    if "last_raw_ocr" not in st.session_state:
        st.session_state["last_raw_ocr"] = ""
    if "pending_text" not in st.session_state:
        st.session_state["pending_text"] = ""
    if "current_text_input" not in st.session_state:
        st.session_state["current_text_input"] = ""
    if "current_len_input" not in st.session_state:
        st.session_state["current_len_input"] = 3    # default spinner

    # Apply pending OCR update BEFORE widgets are created
    if st.session_state["pending_text"]:
        st.session_state["current_text_input"] = st.session_state["pending_text"]
        st.session_state["current_text"] = st.session_state["pending_text"]
        st.session_state["pending_text"] = ""

    # Load model early (only slow first time)
    with st.spinner("Laddar OCR-modell (första gången kan ta en stund)..."):
        _ = load_trocr_model()

    # --- Top row: title + upload / controls, all in one row ---

    col_title, col_file, col_switch, col_reset = st.columns([4, 3, 2, 2])

    with col_title:
        st.title("Korsordslösaren")

    with col_file:
        if st.session_state["image"] is None:
            uploaded = st.file_uploader(
                "Lägg till korsordsbild",
                type=["jpg", "jpeg", "png"],
            )
            if uploaded is not None:
                st.session_state["image"] = uploaded
                st.session_state["image_name"] = uploaded.name
                st.session_state["current_text"] = ""
                st.session_state["current_len"] = 3
                st.session_state["saved_queries"] = []
                st.session_state["last_raw_ocr"] = ""
                st.session_state["current_text_input"] = ""
                st.session_state["current_len_input"] = 3
                st.rerun()
        else:
            name = st.session_state["image_name"] or "Okänt filnamn"
            st.markdown(f"**{name} vald**")

    with col_switch:
        if st.session_state["image"] is not None:
            if st.button("Byt korsord"):
                st.session_state["image"] = None
                st.session_state["image_name"] = ""
                st.session_state["current_text"] = ""
                st.session_state["current_len"] = 3
                st.session_state["last_raw_ocr"] = ""
                st.session_state["current_text_input"] = ""
                st.session_state["current_len_input"] = 3
                st.rerun()

    with col_reset:
        if st.session_state["image"] is not None or st.session_state["saved_queries"]:
            if st.button("Nollställ"):
                st.session_state["confirm_reset"] = True

    # Help text under upload when no image
    if st.session_state["image"] is None and not st.session_state["confirm_reset"]:
        st.write("Välj en korsordsbild för att komma igång.")

    # Confirm reset dialog
    if st.session_state["confirm_reset"]:
        st.warning(
            "Är du säker på att du vill nollställa bilden och alla sparade sökningar?"
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Ja, nollställ", key="reset_yes"):
                st.session_state.clear()
                st.rerun()
        with c2:
            if st.button("Avbryt", key="reset_no"):
                st.session_state["confirm_reset"] = False

    # If still no image after possible reset → stop
    if st.session_state.get("image") is None:
        return

    # Read full-resolution image
    uploaded_file = st.session_state["image"]
    pil_img_full, img_bgr_full = read_pil_and_bgr(uploaded_file)

    # Use full-res for cropper (sharp text)
    pil_img_crop = pil_img_full
    img_bgr = cv2.cvtColor(np.array(pil_img_crop), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    # ---- Two-column layout: cropper left, preview & OCR right ----

    col_left, col_right = st.columns([3, 2])

    # LEFT: image + cropper
    with col_left:
        st.subheader("Markera korsordsområde / ledtråd")

        if HAS_CROPPER:
            st.caption("Dra i rutan på bilden. Dubbelklicka för att låsa beskärningen.")
            cropped_pil = st_cropper(
                pil_img_crop,
                realtime_update=True,
                box_color="#FF0000",
            )
            cropped_bgr = cv2.cvtColor(np.array(cropped_pil), cv2.COLOR_RGB2BGR)
        else:
            st.warning(
                "streamlit-cropper är inte installerat – använder sliders i stället.\n"
                "Installera med `pip install streamlit-cropper` för bättre beskärning."
            )
            st.image(pil_img_crop, use_container_width=True)

            x1 = st.slider("Vänster (x1)", 0, w - 1, 0)
            x2 = st.slider("Höger (x2)", 1, w, w)
            y1 = st.slider("Topp (y1)", 0, h - 1, 0)
            y2 = st.slider("Botten (y2)", 1, h, h)
            cropped_bgr = crop_image_with_box(img_bgr, x1, y1, x2, y2)

    # RIGHT: preview + compact OCR + save row
    with col_right:
        st.subheader("Valt område & OCR")

        ch, cw = cropped_bgr.shape[:2]
        st.caption(f"Område som OCR körs på ({cw}×{ch} px)")

        st.image(
            cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB),
            channels="RGB",
            use_container_width=True,
        )

        MAX_OCR_SIDE = 2600
        max_side = max(ch, cw)
        if max_side > MAX_OCR_SIDE:
            st.warning(
                f"Området är stort ({cw}×{ch} px). OCR kan ta längre tid – "
                "försök gärna markera en mindre ruta om det går."
            )

        # --- Compact row: Text | Len | 🔍 | 💾 ---

        st.markdown("**Aktuell ledtråd**")
        row1, row2, row3, row4 = st.columns([6, 1, 1, 1])

        with row1:
            clue_text = st.text_input(
                "Text",
                key="current_text_input",
                label_visibility="collapsed",
                placeholder="Ledtråd här …",
            )
            st.session_state["current_text"] = clue_text

        with row2:
            length_val = st.number_input(
                "Antal bokstäver",
                min_value=0,
                max_value=30,
                step=1,
                key="current_len_input",
                label_visibility="collapsed",
            )
            st.session_state["current_len"] = int(length_val)

        with row3:
            analyze_clicked = st.button("🔍", help="Analysera med OCR")

        with row4:
            save_clicked = st.button("💾", help="Spara sökfråga")

        # --- Analyse button: run OCR, push text into pending, then rerun ---

        if analyze_clicked:
            with st.spinner("Kör TrOCR och försöker läsa ledtråden..."):
                raw_text = ocr_box_auto(cropped_bgr, max_lines=4)
                clue = detect_clue(raw_text)

            st.session_state["last_raw_ocr"] = raw_text

            if not clue:
                st.error(
                    "Ingen rimlig ledtråd hittades. "
                    "Prova att zooma in mer, flytta rutan eller ta en skarpare bild."
                )
            else:
                # Only update text, NOT length
                st.session_state["pending_text"] = clue
                st.rerun()

        # Raw OCR info
        if st.session_state["last_raw_ocr"]:
            st.caption(f"OCR-resultat (rå): `{st.session_state['last_raw_ocr']}`")

        # Save button
        if save_clicked and st.session_state["current_text"]:
            st.session_state["saved_queries"].append(
                {
                    "text": st.session_state["current_text"],
                    "length": int(st.session_state["current_len"]),
                }
            )
            st.success("Sparad sökfråga.")

    # ---- Saved search queries ----

    saved = st.session_state["saved_queries"]
    if saved:
        st.subheader("Sparade sökningar")
        st.write("Din lista med ledtrådar du vill googla:")

        for j, q in enumerate(saved, start=1):
            text = q["text"]
            length = q["length"]
            url = google_search_url_for_clue(text, length)
            length_info = f"{length} bokstäver" if length > 0 else "längd okänd"

            st.markdown(
                f"**{j}.** `{text}` – {length_info}  \n"
                f"[Sök på Google]({url})"
            )

    # ---- Help text at the bottom ----
    st.markdown("---")
    st.caption(
        """Använder TrOCR (transformer-OCR) för att läsa korsordsledtrådar
och skapa klickbara Google-sökningar.

Workflow: ladda upp en bild av hela korsordet, dra rutan på bilden till en
ledtrådsruta, tryck 🔍 för att läsa av (endast text fylls i), justera texten
vid behov, ställ själv in antal bokstäver (antal rutor) och spara som
sökfråga med 💾."""
    )


if __name__ == "__main__":
    main()

# end of epic program and stuff new update