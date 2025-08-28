from flask import Flask, render_template, request, url_for
import os, time, random
from PIL import Image
import torch
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    AutoTokenizer, AutoModelForSeq2SeqLM
)
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "webp"}
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB

device = "cuda" if torch.cuda.is_available() else "cpu"

# BLIP model for descriptions
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

# FLAN-T5 model for creative rephrasing
tokenizer_style = AutoTokenizer.from_pretrained("google/flan-t5-large")
model_style = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to(device)

# Emojis list
emojis = ["😊", "😂", "😍", "🌸", "🔥", "💖", "✨", "🌞", "😎", "💯", "🥳", "🎉", "🎂"]

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

def generate_caption(image):
    """Generate plain BLIP caption (description)"""
    inputs = processor(images=image, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def style_caption(base_caption, num_variations=3):
    """Generate multiple fun/creative captions"""
    variations = []
    for _ in range(num_variations):
        prompt = f"""
        Rewrite this description into a short, fun, poetic Instagram caption. 
        Add emojis, hashtags, and creativity. Avoid sounding like plain description.

        Description: "{base_caption}"
        Creative caption:
        """
        inputs = tokenizer_style(prompt, return_tensors="pt").to(device)
        out = model_style.generate(
            **inputs,
            max_length=50,
            do_sample=True,
            temperature=random.uniform(1.2, 1.8),  # randomness high
            top_p=0.95,
            top_k=50,
            num_return_sequences=1,
        )
        caption = tokenizer_style.decode(out[0], skip_special_tokens=True)

        # Add random emojis
        chosen_emojis = " ".join(random.sample(emojis, k=random.randint(2, 4)))
        variations.append(f"{caption} {chosen_emojis}")
    return variations

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return render_template("index.html", captions=["⚠ No file uploaded"])

    file = request.files["image"]
    if file.filename == "":
        return render_template("index.html", captions=["⚠ No file selected"])
    if not allowed_file(file.filename):
        return render_template("index.html", captions=["⚠ File type not allowed"])

    try:
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        image = Image.open(filepath).convert("RGB")

        # BLIP description
        base_caption = generate_caption(image)

        # Multiple fun captions
        captions = style_caption(base_caption, num_variations=3)

        return render_template(
            "index.html",
            uploaded_image=url_for("static", filename=f"uploads/{filename}"),
            captions=captions,
        )

    except Exception as e:
        return render_template("index.html", captions=[f"⚠ Error: {str(e)}"])

if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    print("✓ Server running at http://127.0.0.1:5000")
    app.run(debug=True, use_reloader=False)
