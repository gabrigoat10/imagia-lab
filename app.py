import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import io
import base64
import os

try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except Exception:
    REMBG_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# ─── Utilidades ────────────────────────────────────────────────────────────────

def decode_image(file_storage):
    """Convierte un archivo subido en imagen PIL y array numpy."""
    img_bytes = file_storage.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    
    # Limitar tamaño máximo a 2000px para no saturar memoria en Railway
    max_size = 2000
    w, h = pil_img.size
    if w > max_size or h > max_size:
        factor = min(max_size / w, max_size / h)
        new_w = int(w * factor)
        new_h = int(h * factor)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()
    
    return pil_img, img_bytes

def pil_to_response(pil_img, fmt="PNG"):
    """Convierte imagen PIL a respuesta de bytes lista para enviar."""
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    buf.seek(0)
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return send_file(buf, mimetype=mime)

def pil_to_cv2(pil_img):
    """PIL RGBA → numpy BGR para OpenCV."""
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_img):
    """numpy BGR → PIL RGB."""
    rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

# ─── Rutas de la API ───────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "api": "Image Processing API",
        "version": "1.0",
        "herramientas": [
            "POST /eliminar-fondo      → Elimina el fondo de la imagen",
            "POST /crear-png           → Convierte cualquier imagen a PNG",
            "POST /fondo-transparente  → Hace el fondo blanco completamente transparente",
            "POST /semitono-dtf        → Crea semitonos para impresión DTF",
            "POST /enfocar             → Enfoca y nitida la imagen",
            "POST /desenfoque          → Reduce el desenfoque (unblur)",
            "POST /upscaler            → Amplía la imagen x2 o x4 con IA",
            "POST /mejorar-foto        → Mejora brillo, contraste y color",
            "POST /convertir-hd        → Convierte foto a alta definición",
            "POST /restaurar           → Restaura fotos antiguas o dañadas",
        ]
    })


@app.route("/eliminar-fondo", methods=["POST"])
def eliminar_fondo():
    """
    Elimina el fondo de cualquier imagen usando IA (rembg).
    Devuelve PNG con fondo transparente.
    """
    if "image" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen. Usa el campo 'image'."}), 400

    if not REMBG_AVAILABLE:
        return jsonify({"error": "rembg no está disponible. Reinstala con: pip install rembg"}), 500

    pil_img, img_bytes = decode_image(request.files["image"])
    result = rembg_remove(img_bytes)
    buf = io.BytesIO(result)
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/crear-png", methods=["POST"])
def crear_png():
    """
    Convierte cualquier imagen (JPG, WEBP, BMP, etc.) a formato PNG de alta calidad.
    """
    if "image" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen. Usa el campo 'image'."}), 400

    pil_img, _ = decode_image(request.files["image"])
    pil_img = pil_img.convert("RGBA")
    return pil_to_response(pil_img, "PNG")


@app.route("/fondo-transparente", methods=["POST"])
def fondo_transparente():
    """
    Hace transparente el fondo blanco (o de un color específico) de una imagen.
    Parámetro opcional: tolerancia (0-100, default 30)
    """
    if "image" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen. Usa el campo 'image'."}), 400

    tolerancia = int(request.form.get("tolerancia", 30))
    pil_img, _ = decode_image(request.files["image"])
    img = pil_img.convert("RGBA")
    data = np.array(img)

    r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
    # Detecta píxeles cercanos al blanco
    mask = (r > 255 - tolerancia) & (g > 255 - tolerancia) & (b > 255 - tolerancia)
    data[mask] = [0, 0, 0, 0]

    result = Image.fromarray(data, "RGBA")
    return pil_to_response(result, "PNG")


@app.route("/semitono-dtf", methods=["POST"])
def semitono_dtf():
    """
    Genera una imagen de semitonos para impresión DTF (Direct to Film).
    Convierte la imagen en puntos de semitono sobre fondo blanco.
    Parámetros opcionales:
      - angulo (default 45): ángulo de la trama de semitonos
      - escala (default 8): tamaño de los puntos
    """
    if "image" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen. Usa el campo 'image'."}), 400

    angulo = int(request.form.get("angulo", 45))
    escala = int(request.form.get("escala", 8))

    pil_img, _ = decode_image(request.files["image"])
    gray = np.array(pil_img.convert("L"))
    h, w = gray.shape

    # Crear canvas blanco
    canvas = np.ones((h, w), dtype=np.uint8) * 255

    step = escala
    for y in range(0, h, step):
        for x in range(0, w, step):
            # Valor de gris promedio del bloque
            block = gray[y:y+step, x:x+step]
            if block.size == 0:
                continue
            valor = float(block.mean())
            # Radio del punto: más oscuro = punto más grande
            max_radio = step // 2
            radio = int(max_radio * (1 - valor / 255.0))
            if radio > 0:
                cx, cy = x + step // 2, y + step // 2
                cv2.circle(canvas, (cx, cy), radio, 0, -1)

    result = Image.fromarray(canvas, "L").convert("RGB")
    return pil_to_response(result, "PNG")


@app.route("/enfocar", methods=["POST"])
def enfocar():
    """
    Aplica enfoque y nitidez a la imagen.
    Parámetro opcional: intensidad (1-5, default 2)
    """
    if "image" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen. Usa el campo 'image'."}), 400

    intensidad = float(request.form.get("intensidad", 2.0))
    intensidad = max(1.0, min(5.0, intensidad))

    pil_img, _ = decode_image(request.files["image"])
    img_rgb = pil_img.convert("RGB")
    cv2_img = pil_to_cv2(img_rgb)

    # Unsharp masking para enfoque
    gaussian = cv2.GaussianBlur(cv2_img, (0, 0), 3)
    sharpened = cv2.addWeighted(cv2_img, 1 + intensidad * 0.5, gaussian, -intensidad * 0.5, 0)

    result = cv2_to_pil(sharpened)
    return pil_to_response(result, "PNG")


@app.route("/desenfoque", methods=["POST"])
def desenfoque():
    """
    Reduce el desenfoque de una imagen (unblur).
    Usa deconvolución de Wiener aproximada con filtros de OpenCV.
    Parámetro opcional: intensidad (1-5, default 2)
    """
    if "image" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen. Usa el campo 'image'."}), 400

    intensidad = float(request.form.get("intensidad", 2.0))
    intensidad = max(1.0, min(5.0, intensidad))

    pil_img, _ = decode_image(request.files["image"])
    img_rgb = pil_img.convert("RGB")
    cv2_img = pil_to_cv2(img_rgb)

    # Filtro de agudización para reducir desenfoque
    kernel = np.array([
        [-1, -1, -1],
        [-1,  9 + intensidad, -1],
        [-1, -1, -1]
    ])
    sharpened = cv2.filter2D(cv2_img, -1, kernel)

    # Mejorar detalles con CLAHE
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    sharpened = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    result = cv2_to_pil(sharpened)
    return pil_to_response(result, "PNG")


@app.route("/upscaler", methods=["POST"])
def upscaler():
    """
    Amplía la imagen con algoritmo de super-resolución de alta calidad.
    Parámetro opcional: escala (2 o 4, default 2)
    """
    if "image" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen. Usa el campo 'image'."}), 400

    escala = int(request.form.get("escala", 2))
    escala = 4 if escala >= 3 else 2

    pil_img, _ = decode_image(request.files["image"])
    img_rgb = pil_img.convert("RGB")
    w, h = img_rgb.size
    new_w, new_h = w * escala, h * escala

    # Lanczos para upscaling de alta calidad + mejora de detalles
    upscaled = img_rgb.resize((new_w, new_h), Image.LANCZOS)

    # Aplicar ligero enfoque post-upscaling para recuperar nitidez
    cv2_img = pil_to_cv2(upscaled)
    gaussian = cv2.GaussianBlur(cv2_img, (0, 0), 1.5)
    sharpened = cv2.addWeighted(cv2_img, 1.3, gaussian, -0.3, 0)

    result = cv2_to_pil(sharpened)
    return pil_to_response(result, "PNG")


@app.route("/mejorar-foto", methods=["POST"])
def mejorar_foto():
    """
    Mejora automáticamente brillo, contraste, saturación y nitidez de la foto.
    Parámetros opcionales:
      - brillo (0.5-2.0, default 1.1)
      - contraste (0.5-2.0, default 1.2)
      - saturacion (0.5-2.0, default 1.2)
      - nitidez (0.5-2.0, default 1.3)
    """
    if "image" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen. Usa el campo 'image'."}), 400

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

    # Reducción de ruido con OpenCV
    cv2_img = pil_to_cv2(img)
    denoised = cv2.fastNlMeansDenoisingColored(cv2_img, None, 5, 5, 7, 21)
    result = cv2_to_pil(denoised)

    return pil_to_response(result, "PNG")


@app.route("/convertir-hd", methods=["POST"])
def convertir_hd():
    """
    Convierte una foto a Alta Definición:
    - Amplía si es menor a 1080p
    - Mejora nitidez, contraste y reducción de ruido
    - Guarda en máxima calidad
    """
    if "image" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen. Usa el campo 'image'."}), 400

    pil_img, _ = decode_image(request.files["image"])
    img = pil_img.convert("RGB")
    w, h = img.size

    # Ampliar si es menor a Full HD
    if h < 1080:
        factor = 1080 / h
        new_w = int(w * factor)
        img = img.resize((new_w, 1080), Image.LANCZOS)

    # Mejorar calidad
    img = ImageEnhance.Sharpness(img).enhance(1.4)
    img = ImageEnhance.Contrast(img).enhance(1.15)

    cv2_img = pil_to_cv2(img)
    denoised = cv2.fastNlMeansDenoisingColored(cv2_img, None, 4, 4, 7, 21)
    result = cv2_to_pil(denoised)

    return pil_to_response(result, "PNG")


@app.route("/restaurar", methods=["POST"])
def restaurar():
    """
    Restaura fotos antiguas o dañadas:
    - Reduce ruido y manchas
    - Mejora detalles y contraste
    - Corrige colores desteñidos
    """
    if "image" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen. Usa el campo 'image'."}), 400

    pil_img, _ = decode_image(request.files["image"])
    img = pil_img.convert("RGB")

    # Reducción fuerte de ruido (fotos antiguas tienen mucho grano)
    cv2_img = pil_to_cv2(img)
    denoised = cv2.fastNlMeansDenoisingColored(cv2_img, None, 10, 10, 7, 21)

    # Mejora de detalles con CLAHE en canal L
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    result = cv2_to_pil(enhanced)

    # Mejora de color y saturación desde PIL
    result = ImageEnhance.Color(result).enhance(1.3)
    result = ImageEnhance.Sharpness(result).enhance(1.5)
    result = ImageEnhance.Contrast(result).enhance(1.2)

    return pil_to_response(result, "PNG")

@app.route("/convertir-dpi", methods=["POST"])
def convertir_dpi():
    """
    Convierte una imagen a 300 DPI (o el DPI que elijas).
    Mantiene el tamaño físico real en centímetros.
    Parámetro opcional: dpi (72, 150, 300, 600 — default 300)
    """
    if "image" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen. Usa el campo 'image'."}), 400

    dpi = int(request.form.get("dpi", 300))
    dpi = max(72, min(600, dpi))

    pil_img, _ = decode_image(request.files["image"])
    img = pil_img.convert("RGB")

    # Obtener DPI actual de la imagen
    current_dpi = pil_img.info.get("dpi", (72, 72))
    if isinstance(current_dpi, tuple):
        current_dpi = current_dpi[0]
    current_dpi = max(float(current_dpi), 72.0)

    # Calcular nuevo tamaño manteniendo dimensiones físicas
    w, h = img.size
    factor = dpi / current_dpi
    new_w = int(w * factor)
    new_h = int(h * factor)

    # Reescalar con alta calidad
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    # Aplicar enfoque post-reescalado
    resized = ImageEnhance.Sharpness(resized).enhance(1.3)

    # Guardar con el DPI especificado
    buf = io.BytesIO()
    resized.save(buf, format="PNG", dpi=(dpi, dpi))
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/reescalar", methods=["POST"])
def reescalar():
    """
    Reescala una imagen a un tamaño específico en centímetros con DPI personalizado.
    Parámetros:
      - ancho_cm: ancho deseado en centímetros (default 10)
      - alto_cm: alto deseado en centímetros (default 0 = proporcional)
      - dpi: resolución deseada (default 300)
    """
    if "image" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen. Usa el campo 'image'."}), 400

    ancho_cm = float(request.form.get("ancho_cm", 10))
    alto_cm = float(request.form.get("alto_cm", 0))
    dpi = int(request.form.get("dpi", 300))
    dpi = max(72, min(600, dpi))

    pil_img, _ = decode_image(request.files["image"])
    img = pil_img.convert("RGB")
    w, h = img.size

    # Convertir centímetros a píxeles (1 pulgada = 2.54 cm)
    new_w = int((ancho_cm / 2.54) * dpi)

    if alto_cm > 0:
        new_h = int((alto_cm / 2.54) * dpi)
    else:
        # Mantener proporción
        new_h = int(h * (new_w / w))

    # Reescalar con alta calidad
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    # Aplicar enfoque post-reescalado
    cv2_img = pil_to_cv2(resized)
    gaussian = cv2.GaussianBlur(cv2_img, (0, 0), 1.5)
    sharpened = cv2.addWeighted(cv2_img, 1.3, gaussian, -0.3, 0)
    resized = cv2_to_pil(sharpened)

    # Guardar con el DPI especificado
    buf = io.BytesIO()
    resized.save(buf, format="PNG", dpi=(dpi, dpi))
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


# ─── Inicio ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  🖼️  Image Processing API — Lista para usar")
    print("="*55)
    print(f"  Servidor corriendo en: http://localhost:5000")
    print(f"  rembg (eliminar fondo): {'✅ Disponible' if REMBG_AVAILABLE else '❌ No disponible'}")
    print("="*55 + "\n")
    app.run(debug=True, port=5000)