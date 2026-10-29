from flask import Flask, request, render_template
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

app = Flask(__name__)

MODEL_NAME = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()


def generate_description(product_name, keywords):
    prompt = (
        "Task: Write a useful online store product description. "
        "Write exactly two complete sentences. "
        f"Product name: {product_name}. "
        f"Required keywords: {keywords}. "
        "Explain how the product may be used in everyday life. "
        "Write the actual description, not a summary of the inputs. "
        "Do not start with 'Product:' or 'Keywords:'. "
        "Do not invent features or specifications."
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            min_new_tokens=25,
            num_beams=5,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2
        )

    description = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()

    return description


@app.route("/", methods=["GET", "POST"])
def home():
    product_name = ""
    keywords = ""
    description = None
    error = None

    if request.method == "POST":
        product_name = request.form.get("product_name", "").strip()
        keywords = request.form.get("keywords", "").strip()

        if not product_name or not keywords:
            error = "Please enter both a product name and keywords."
        else:
            try:
                description = generate_description(
                    product_name, keywords
                )
            except Exception:
                app.logger.exception("Generation failed")
                error = "Description generation failed. Please try again."

    return render_template(
        "index.html",
        product_name=product_name,
        keywords=keywords,
        description=description,
        error=error
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
