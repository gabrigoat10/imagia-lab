import os
import io
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io

try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except Exception:
    REMBG_AVAILABLE = False

app = Flask(__name__)
CORS(app)

MAX_INPUT = 1800
MAX_OUTPUT = 1200

def decode_image(file_storage):
    img_bytes = file_storage.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    w, h = pil_img.size
    if w > MAX_INPUT or h > MAX_INPUT:
        factor = min(MAX_INPUT / w, MAX_INPUT / h)
        pil_img = pil_img.resize((int(w * factor), int(h * factor)), Image.LANCZOS)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()
    return pil_img, img_bytes

def pil_to_response(pil_img, fmt="PNG", dpi=None):
    w, h = pil_img.size
    if w > MAX_OUTPUT or h > MAX_OUTPUT:
        factor = min(MAX_OUTPUT / w, MAX_OUTPUT / h)
        pil_img = pil_img.resize((int(w * factor), int(h * factor)), Image.LANCZOS)
    buf = io.BytesIO()
    save_kwargs = {"format": "PNG", "optimize": True}
    if dpi:
        save_kwargs["dpi"] = (dpi, dpi)
    pil_img.save(buf, **save_kwargs)
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

def pil_to_cv2(pil_img):
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_img):
    rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "api": "ImagIA Lab API",
        "version": "2.0",
        "estado": "activa",
        "herramientas": [
            "POST /eliminar-fondo",
            "POST /crear-png",
            "POST /fondo-transparente",
            "POST /semitono-dtf",
            "POST /enfocar",
            "POST /desenfoque",
            "POST /upscaler",
            "POST /mejorar-foto",
            "POST /convertir-hd",
            "POST /restaurar",
            "POST /convertir-dpi",
            "POST /reescalar",
        ]
    })

@app.route("/eliminar-fondo", methods=["POST"])
def eliminar_fondo():
    if "image" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen."}), 400
    if not REMBG_AVAILABLE:
        return jsonify({"error": "rembg no disponible."}), 500
    pil_img, img_bytes = decode_image(request.files["image"])
    result = rembg_remove(img_bytes)
    buf = io.BytesIO(result)
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

@app.route("/crear-png", methods=["POST"])
def crear_png():
    if "image" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen."}), 400
    pil_img, _ = decode_image(request.files["image"])
    return pil_to_response(pil_img.convert("RGBA"))

@app.route("/fondo-transparente", methods=["POST"])
def fondo_transparente():
    if "image" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen."}), 400
    tolerancia = int(request.form.get("tolerancia", 30))
    pil_img, _ = decode_image(request.files["image"])
    img = pil_img.convert("RGBA")
    data = np.array(img)
    r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
    mask = (r > 255 - tolerancia) & (g > 255 - tolerancia) & (b > 255 - tolerancia)
    data[mask] = [0, 0, 0, 0]
    return pil_to_response(Image.fromarray(data, "RGBA"))

@app.route("/semitono-dtf", methods=["POST"])
def semitono_dtf():
    if "image" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen."}), 400
    escala = int(request.form.get("escala", 8))
    pil_img, _ = decode_image(request.files["image"])
    gray = np.array(pil_img.convert("L"))
    h, w = gray.shape
    canvas = np.ones((h, w), dtype=np.uint8) * 255
    step = escala
    for y in range(0, h, step):
        for x in range(0, w, step):
            block = gray[y:y+step, x:x+step]
            if block.size == 0:
                continue
            valor = float(block.mean())
            radio = int((step // 2) * (1 - valor / 255.0))
            if radio > 0:
                cv2.circle(canvas, (x + step // 2, y + step // 2), radio, 0, -1)
    return pil_to_response(Image.fromarray(canvas, "L").convert("RGB"))

@app.route("/enfocar", methods=["POST"])
def enfocar():
    if "image" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen."}), 400
    intensidad = max(1.0, min(5.0, float(request.form.get("intensidad", 2.0))))
    pil_img, _ = decode_image(request.files["image"])
    cv2_img = pil_to_cv2(pil_img)
    gaussian = cv2.GaussianBlur(cv2_img, (0, 0), 3)
    sharpened = cv2.addWeighted(cv2_img, 1 + intensidad * 0.5, gaussian, -intensidad * 0.5, 0)
    return pil_to_response(cv2_to_pil(sharpened))

@app.route("/desenfoque", methods=["POST"])
def desenfoque():
    if "image" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen."}), 400
    intensidad = max(1.0, min(5.0, float(request.form.get("intensidad", 2.0))))
    pil_img, _ = decode_image(request.files["image"])
    cv2_img = pil_to_cv2(pil_img)
    kernel = np.array([[-1,-1,-1],[-1, 9 + intensidad,-1],[-1,-1,-1]])
    sharpened = cv2.filter2D(cv2_img, -1, kernel)
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    sharpened = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)
    return pil_to_response(cv2_to_pil(sharpened))

@app.route("/upscaler", methods=["POST"])
def upscaler():
    if "image" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen."}), 400
    escala = 4 if int(request.form.get("escala", 2)) >= 3 else 2
    pil_img, _ = decode_image(request.files["image"])
    img_rgb = pil_img.convert("RGB")
    w, h = img_rgb.size
    upscaled = img_rgb.resize((w * escala, h * escala), Image.LANCZOS)
    cv2_img = pil_to_cv2(upscaled)
    gaussian = cv2.GaussianBlur(cv2_img, (0, 0), 1.5)
    sharpened = cv2.addWeighted(cv2_img, 1.3, gaussian, -0.3, 0)
    return pil_to_response(cv2_to_pil(sharpened))

@app.route("/mejorar-foto", methods=["POST"])
def mejorar_foto():
    if "image" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen."}), 400
    brillo = float(request.form.get("brillo", 1.1))
    contraste = float(request.form.get("contraste", 1.2))
    saturacion = float(request.form.get("saturacion", 1.2))
    nitidez = float(request.form.get("nitidez", 1.3))
    pil_img, _ = decode_image(request.files["image"])
    img = pil_img.convert("RGB")
    img = ImageEnhance.Brightness(img).enhance(brillo)
    img = ImageEnhance.Contrast(img).enhance(contraste)
    img = ImageEnhance.Color(img).enhance(saturacion)
    img = ImageEnhance.Sharpness(img).enhance(nitidez)
    cv2_img = pil_to_cv2(img)
    denoised = cv2.fastNlMeansDenoisingColored(cv2_img, None, 5, 5, 7, 21)
    return pil_to_response(cv2_to_pil(denoised))

@app.route("/convertir-hd", methods=["POST"])
def convertir_hd():
    if "image" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen."}), 400
    pil_img, _ = decode_image(request.files["image"])
    img = pil_img.convert("RGB")
    w, h = img.size
    if h < 1080:
        factor = 1080 / h
        img = img.resize((int(w * factor), 1080), Image.LANCZOS)
    img = ImageEnhance.Sharpness(img).enhance(1.4)
    img = ImageEnhance.Contrast(img).enhance(1.15)
    cv2_img = pil_to_cv2(img)
    denoised = cv2.fastNlMeansDenoisingColored(cv2_img, None, 4, 4, 7, 21)
    return pil_to_response(cv2_to_pil(denoised))

@app.route("/restaurar", methods=["POST"])
def restaurar():
    if "image" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen."}), 400
    pil_img, _ = decode_image(request.files["image"])
    cv2_img = pil_to_cv2(pil_img.convert("RGB"))
    denoised = cv2.fastNlMeansDenoisingColored(cv2_img, None, 10, 10, 7, 21)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)
    result = cv2_to_pil(enhanced)
    result = ImageEnhance.Color(result).enhance(1.3)
    result = ImageEnhance.Sharpness(result).enhance(1.5)
    result = ImageEnhance.Contrast(result).enhance(1.2)
    return pil_to_response(result)

@app.route("/convertir-dpi", methods=["POST"])
def convertir_dpi():
    if "image" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen."}), 400
    dpi = max(72, min(600, int(request.form.get("dpi", 300))))
    pil_img, _ = decode_image(request.files["image"])
    img = pil_img.convert("RGB")
    current_dpi = max(float(pil_img.info.get("dpi", (72, 72))[0]), 72.0)
    w, h = img.size
    factor = dpi / current_dpi
    resized = img.resize((int(w * factor), int(h * factor)), Image.LANCZOS)
    resized = ImageEnhance.Sharpness(resized).enhance(1.3)
    return pil_to_response(resized, dpi=dpi)

@app.route("/reescalar", methods=["POST"])
def reescalar():
    if "image" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen."}), 400
    ancho_cm = float(request.form.get("ancho_cm", 10))
    alto_cm = float(request.form.get("alto_cm", 0))
    dpi = max(72, min(600, int(request.form.get("dpi", 300))))
    pil_img, _ = decode_image(request.files["image"])
    img = pil_img.convert("RGB")
    w, h = img.size
    new_w = int((ancho_cm / 2.54) * dpi)
    new_h = int((alto_cm / 2.54) * dpi) if alto_cm > 0 else int(h * (new_w / w))
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    cv2_img = pil_to_cv2(resized)
    gaussian = cv2.GaussianBlur(cv2_img, (0, 0), 1.5)
    sharpened = cv2.addWeighted(cv2_img, 1.3, gaussian, -0.3, 0)
    return pil_to_response(cv2_to_pil(sharpened), dpi=dpi)

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  ImagIA Lab API v2.0 — Lista para usar")
    print("="*55)
    print(f"  Servidor: http://localhost:5000")
    print(f"  rembg: {'✅ Disponible' if REMBG_AVAILABLE else '❌ No disponible'}")
    print("="*55 + "\n")
    app.run(debug=False, host='0.0.0.0', port=5000)
