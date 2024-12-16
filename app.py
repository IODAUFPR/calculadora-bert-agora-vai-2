from flask import Flask, render_template, request, jsonify
from transformers import BertTokenizer, BertModel
import torch
from scipy.spatial.distance import cosine

# Inicializando Flask
app = Flask(__name__)

# Carregando o modelo BERT e o tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def get_bert_embedding(text):
    """Obtém o embedding BERT para o texto fornecido."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embeddings


def bert_similarity(s1, s2):
    """Calcula a similaridade entre dois textos usando embeddings do BERT."""
    emb1 = get_bert_embedding(s1)
    emb2 = get_bert_embedding(s2)
    similarity = 1 - cosine(emb1, emb2)
    return similarity * 100


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/calculate", methods=["POST"])
def calculate():
    try:
        # Recuperar os dados do formulário
        marca = request.form.get("marca").strip()
        colidencias_text = request.form.get("colidencias").strip()

        if not marca or not colidencias_text:
            return jsonify({"error": "Preencha todos os campos antes de continuar."}), 400

        colidencias = colidencias_text.split("\n")
        results = []

        # Calcular similaridade para cada marca colidente
        for colidencia in colidencias:
            similarity = bert_similarity(marca, colidencia.strip())
            results.append({"marca": colidencia.strip(), "similarity": f"{similarity:.2f}%"})

        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
