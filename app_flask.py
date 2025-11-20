import os, csv, time
from pathlib import Path
from flask import Flask, request, render_template, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from matcher import query_image  


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

BASE = Path(__file__).resolve().parent
APP = Flask(__name__, template_folder=str(BASE/"templates"), static_folder=str(BASE/"static"))

MODEL = BASE/"model"
REF   = BASE/"images"/"reference"
UP    = BASE/"data"/"uploads"
OUT   = BASE/"output"
UP.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

LOG = OUT/"manual_verifications.csv"
if not LOG.exists():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(["timestamp","query_abs","reference_abs","score"])

@APP.route("/reference/<path:filename>")
def serve_reference(filename):
    return send_from_directory(REF, filename)

@APP.route("/uploads/<path:filename>")
def serve_uploads(filename):
    return send_from_directory(UP, filename)

@APP.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@APP.route("/search", methods=["POST"])
def search():
    f = request.files.get("image")
    if not f or f.filename == "":
        return jsonify({"error":"no file"}), 400

    fname = secure_filename(f.filename)
    save = UP / fname
    f.save(str(save))

    topk = int(request.form.get("topk", 6))
    results = query_image(str(MODEL), str(save), topk=topk, show=False)  # list[(abs_path, score)]

    payload = {
        "query": fname,
        "results": [
            {"path": str(Path(p).resolve()),
             "basename": Path(p).name,
             "score": float(s)} for (p, s) in results
        ]
    }
    return jsonify(payload)

@APP.route("/verify", methods=["POST"])
def verify():
    data = request.get_json(silent=True) or request.form
    q = data.get("query")
    ref = data.get("reference_abs", "NEW")
    sc  = float(data.get("score", 0.0))
    qabs = (UP/q).resolve() if q else ""

    with open(LOG, "a", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow([int(time.time()), str(qabs), str(ref), sc])

    return jsonify({"status":"ok"})

if __name__=="__main__":
    APP.run(host="0.0.0.0", port=5000, debug=True)
    