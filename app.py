from flask import Flask, render_template, request
from transformers import BartTokenizer, BartForConditionalGeneration

app = Flask(__name__)

def summarize_text(text, max_length=100, num_beams=4):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=1024)

    summary_ids = model.generate(
        input_ids,
        max_length=max_length,
        num_beams=num_beams,
        no_repeat_ngram_size=2,
        early_stopping=True,
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form["text"]
        max_length = int(request.form["max_length"])
        num_beams = int(request.form["num_beams"])
        summary = summarize_text(text, max_length, num_beams)
        return render_template("index.html", summary=summary)
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True) 

    #nothing just for fun