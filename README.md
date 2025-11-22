# Crossword Helper – Prototype

A local Streamlit app that lets you:

1. Upload a crossword image
2. Select the crossword area (optional – default is entire image)
3. Run OCR on the selected area
4. Treat each OCR line as a clue
5. Generate clickable Google search links like: `korsord Bonde som piper 3 bokstäver`

## 1. Prerequisites (macOS)

### Install Homebrew (if you don't have it)

See https://brew.sh/ for the exact command.

### Install Python 3 (if needed)

macOS usually has Python, but it's best to install a recent one:

```bash
brew install python
```

### Install Tesseract OCR

```bash
brew install tesseract
```

For Swedish language support (optional but recommended):

```bash
brew install tesseract-lang
```

Make sure `tesseract` is on your PATH (usually via Homebrew).

## 2. Create and activate a virtual environment

From the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the app

```bash
streamlit run app.py
```

Your browser should open at something like `http://localhost:8501`.

## 5. Usage

1. Upload a JPG/PNG of your crossword.
2. If needed, uncheck **"Använd hela bilden som korsord"** and use the sliders to crop only the crossword area.
3. Choose OCR language (`swe` or `eng`) – make sure that language is installed in Tesseract.
4. Click **"Analysera korsord"**.
5. The app will:
   - Run OCR on the cropped area
   - Show each OCR line as a potential clue
   - Let you adjust the clue text and number of letters
   - Provide a Google search link like:

   `korsord <ledtråd> <antal> bokstäver`

This is a prototype. The logic for splitting boxes, arrows, and exact word paths is *not* fully implemented yet – it's set up so you can gradually replace the OCR + clue detection parts with more advanced OpenCV logic.
