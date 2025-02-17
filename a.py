import os
import re
import uuid
import zipfile
import logging
from datetime import datetime

import pdfplumber
import pytesseract
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from flask import Flask, jsonify, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename
import spacy
from invoice2data.extract.loader import read_templates

app = Flask(__name__)
app.config.update({
    'UPLOAD_FOLDER': "uploads",
    'PROCESSED_FOLDER': "processed",
    'TEMP_ZIP': "temp",
    'TEMPLATE_FOLDER': "templates",
    'LOGO_FOLDER': "logos",
    'MAX_CONTENT_LENGTH': 200 * 1024 * 1024  # Max 200MB
})

# Sicherstellen, dass erforderliche Verzeichnisse existieren
for folder in [app.config['UPLOAD_FOLDER'], app.config['PROCESSED_FOLDER'],
               app.config['TEMP_ZIP'], app.config['LOGO_FOLDER'], app.config['TEMPLATE_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('invoice_processor.log'),
        logging.StreamHandler()
    ]
)

# SpaCy-Modell laden
try:
    nlp = spacy.load("de_core_news_sm")
except OSError:
    logging.warning("SpaCy-Modell nicht gefunden. Installation wird gestartet...")
    os.system("python -m spacy download de_core_news_sm")
    nlp = spacy.load("de_core_news_sm")

class InvoiceExtractor:
    def __init__(self, template_folder):
        self.templates = read_templates(template_folder)
        
        # Erweiterte Muster für Rechnungsnummern
        self.invoice_number_patterns = [
            r"(?:Rechnung|Gutschrift|Beleg|Dokument|Faktura)[\s\-]*(?:Nr\.?|Nummer|No\.?)[:\s]*([A-Za-z0-9\-_/]{3,30})",
            r"Rechnungs-?(?:nummer)?[\s:]*([A-Za-z0-9\-_/]{3,30})",
            r"(?:Invoice|Inv\.?)[\s:]*(?:No\.?)?[\s:]*([A-Za-z0-9\-_/]{3,30})",
            r"(?:Dokumentennummer|Belegnummer)[:\s]*([A-Za-z0-9\-_/]{3,30})",
            r"\b(RE\d{6,10})\b",
            r"\b(AR\d{6,10})\b",
            r"\b(ST\d{6,10})\b",
            r"\b(INV-\d{8})\b",
            r"\b(\d{6,12})\b",
        ]
        
        # Muster basierend auf dem Dateinamen
        self.filename_invoice_number_patterns = [
            r'AR_?(AR\d{7})_',
            r'Invoice_(\d{8})',
            r'Kundenrechnung_INV-(\d{8})',
            r'RE(\d{6,8})',
            r'Rechnung_(\d{4}-\d{4,5})',
            r'Rechnung24-(\d{7})',
            r'ST1-\d{5}-(\d{9})',
            r'(\d{6}_\d{3}-\d{8})',
            r'RE_\d{2}_(\d{8})',
            r'Pilz_(\d{8})',
            r'RechnungNr_(\d{7})',
            r'(\d{6})',
        ]

        # Umfassende Datums-Muster
        self.invoice_date_patterns = [
            r'(?:Rechnungs(?:datum|date)|Datum)[\s:]*([0-3]?\d[.\-/][0-1]?\d[.\-/]\d{2,4})',
            r'(?:Ausstellungs(?:datum|date))[\s:]*([0-3]?\d[.\-/][0-1]?\d[.\-/]\d{2,4})',
            r'(?:Leistungs(?:datum|date))[\s:]*([0-3]?\d[.\-/][0-1]?\d[.\-/]\d{2,4})',
            r'(?:vom|Datum)[\s:]*([0-3]?\d[.\-/][0-1]?\d[.\-/]\d{2,4})',
            r'(?:Invoice Date|Date of Issue)[\s:]*([0-3]?\d[.\-/][0-1]?\d[.\-/]\d{2,4})',
            r'\b([0-3]?\d[.\-/][0-1]?\d[.\-/]\d{2,4})\b',
            r'\b(\d{4}[-/]\d{2}[-/]\d{2})\b',
        ]

    def extract_invoice_details(self, text, filename):
        text = text.replace('\n', ' ').replace('\r', '').strip()
        text = re.sub(r'\s+', ' ', text)
        
        invoice_number = self._extract_invoice_number(text, filename)
        invoice_date = self._extract_invoice_date(text)
        
        return {"invoice_number": invoice_number, "invoice_date": invoice_date}

    def _extract_invoice_number(self, text, filename):
        def is_valid_invoice_number(num):
            if not num:
                return False
            if len(num) < 3 or len(num) > 30:
                return False
            if not any(char.isdigit() for char in num):
                return False
            invalid_patterns = [
                r'^\d{1,2}[.,]\d{1,2}[.,]\d{2,4}$',
                r'^\d+[.,]\d{2}$',
                r'^20\d{2}$',
            ]
            return not any(re.match(pattern, num) for pattern in invalid_patterns)
        
        def clean_invoice_number(number):
            number = number.strip()
            number = re.sub(r'^[_\-\.]+|[_\-\.]+$', '', number)
            return number

        for pattern in self.invoice_number_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                number = match.group(1).strip()
                number = clean_invoice_number(number)
                if is_valid_invoice_number(number):
                    logging.info(f"Rechnungsnummer im Text gefunden: {number}")
                    return number

        for pattern in self.filename_invoice_number_patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                number = match.group(1).strip()
                number = clean_invoice_number(number)
                if is_valid_invoice_number(number):
                    logging.info(f"Rechnungsnummer im Dateinamen gefunden: {number}")
                    return number

        logging.error("Keine gültige Rechnungsnummer im Dokument gefunden")
        raise ValueError("Keine gültige Rechnungsnummer im Dokument gefunden")

    def _extract_invoice_date(self, text):
        for pattern in self.invoice_date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for date_str in matches:
                formatted_date = self._normalize_date(date_str)
                if formatted_date:
                    logging.info(f"Rechnungsdatum gefunden: {formatted_date}")
                    return formatted_date
        
        logging.error("Kein gültiges Rechnungsdatum gefunden")
        raise ValueError("Kein gültiges Rechnungsdatum im Dokument gefunden")

    def _normalize_date(self, date_str):
        try:
            date_str = date_str.strip().replace(' ', '')
            separator = None
            for sep in ['.', '-', '/']:
                if sep in date_str:
                    separator = sep
                    break
            if not separator:
                return None
            parts = date_str.split(separator)
            if len(parts) != 3:
                return None

            for part in parts:
                if not part.isdigit():
                    return None

            possible_formats = [
                (0, 1, 2, "%d.%m.%Y"),
                (2, 1, 0, "%Y.%m.%d"),
            ]

            for day_idx, month_idx, year_idx, date_format in possible_formats:
                try:
                    day = int(parts[day_idx])
                    month = int(parts[month_idx])
                    year = parts[year_idx]
                    if len(year) == 2:
                        year = '20' + year if int(year) < 50 else '19' + year
                    year = int(year)
                    if not (1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2100):
                        continue
                    date_obj = datetime(year, month, day)
                    if date_obj > datetime.now():
                        continue
                    return date_obj.strftime("%d.%m.%Y")
                except ValueError:
                    continue
            return None
        except Exception as e:
            logging.error(f"Fehler bei der Datumsnormalisierung: {e}")
            return None

    def preprocess_image(self, image):
        image = image.convert('L')
        image_np = np.array(image)
        denoised = cv2.fastNlMeansDenoising(image_np, h=30)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        coords = np.column_stack(np.where(enhanced > 0))
        angle = 0
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            angle = -(90 + angle) if angle < -45 else -angle
            (h, w) = enhanced.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            enhanced = cv2.warpAffine(enhanced, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        return Image.fromarray(binary)

    def extract_text(self, pdf_path):
        try:
            text = ""
            # Zuerst mit pdfplumber versuchen
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text += " " + page_text.strip()
            
            # Falls kein Text gefunden wurde, OCR verwenden
            if not text.strip():
                logging.info(f"Kein Text gefunden in {pdf_path} mit pdfplumber. Starte OCR.")
                images = convert_from_path(pdf_path, dpi=100)
                for img in images:
                    processed_img = self.preprocess_image(img)
                    page_text = pytesseract.image_to_string(processed_img, lang='deu')
                    if not page_text.strip():
                        page_text = pytesseract.image_to_string(processed_img, lang='eng')
                    text += " " + page_text.strip()
            
            text = ' '.join(text.split())
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        except Exception as e:
            logging.error(f"Fehler beim Extrahieren von Text aus {pdf_path}: {str(e)}")
            return ""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files' not in request.files:
        return jsonify({"error": "No files in request"}), 400

    files = request.files.getlist('files')
    if not files or all(not file.filename for file in files):
        return jsonify({"error": "No files selected"}), 400

    if len(files) > 100:
        return jsonify({"error": "Maximum 100 files can be uploaded at once"}), 400

    extractor = InvoiceExtractor(app.config['TEMPLATE_FOLDER'])
    processed_files = []
    filenames = []
    errors = []

    for file in files:
        if not file.filename:
            continue

        if not file.filename.lower().endswith('.pdf'):
            errors.append(f"{file.filename}: Not a PDF file")
            continue

        try:
            unique_id = uuid.uuid4().hex
            original_filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_{original_filename}")

            file.save(upload_path)

            # Text extrahieren
            text = extractor.extract_text(upload_path)

            try:
                data = extractor.extract_invoice_details(text, original_filename)

                # Neuen Dateinamen mit Rechnungsdetails erstellen
                new_filename = f"Rng.Nr.{data['invoice_number']}_v.{data['invoice_date']}.tiff"
                new_filename = re.sub(r'[\\/*?:"<>|]', "_", new_filename)

                processed_path = os.path.join(app.config['PROCESSED_FOLDER'], new_filename)

                # PDF in Bilder konvertieren – hier wird der DPI reduziert und
                # mit LZW verlustfrei komprimiert, während die Farbinformationen erhalten bleiben.
                images = convert_from_path(upload_path, dpi=150)
                if images:
                    images[0].save(
                        processed_path,
                        format="TIFF",
                        compression="tiff_lzw",
                        save_all=True,
                        append_images=images[1:]
                    )
                    if os.path.exists(processed_path):
                        processed_files.append(processed_path)
                        filenames.append(new_filename)
                        logging.info(f"Erfolgreich verarbeitet: {new_filename}")
                    else:
                        errors.append(f"{original_filename}: Konnte verarbeitete Datei nicht speichern")
                else:
                    errors.append(f"{original_filename}: Konnte PDF nicht in Bilder konvertieren")

            except ValueError as ve:
                errors.append(f"{original_filename}: {str(ve)}")
                logging.error(f"{original_filename}: {str(ve)}")
                continue

        except Exception as e:
            error_msg = f"{original_filename}: {str(e)}"
            errors.append(error_msg)
            logging.error(error_msg)
        finally:
            if 'upload_path' in locals() and os.path.exists(upload_path):
                os.remove(upload_path)

    if not processed_files:
        return jsonify({
            "error": "No files processed successfully",
            "details": errors
        }), 400

    try:
        # ZIP-Datei erstellen
        zip_filename = f"processed_{uuid.uuid4().hex}.zip"
        zip_path = os.path.join(app.config['TEMP_ZIP'], zip_filename)

        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file_path in processed_files:
                if os.path.exists(file_path):
                    zipf.write(file_path, arcname=os.path.basename(file_path))
                    os.remove(file_path)

        download_url = url_for('download_file', filename=zip_filename)
        return jsonify({
            "message": "Files processed successfully",
            "filenames": filenames,
            "download_url": download_url,
            "errors": errors if errors else None
        }), 200

    except Exception as e:
        logging.error(f"Fehler beim Erstellen der ZIP-Datei: {str(e)}")
        return jsonify({
            "error": "Error creating ZIP file",
            "details": str(e)
        }), 500

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    zip_path = os.path.join(app.config['TEMP_ZIP'], filename)
    if os.path.exists(zip_path):
        return send_file(zip_path, as_attachment=True)
    return jsonify({"error": "File not found"}), 404

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
