import os
import sys
import subprocess
from flask import (
    Flask, render_template, request,
    redirect, url_for, send_from_directory, abort
)

BASE = os.path.abspath(os.path.dirname(__file__))

def list_airlines():
    """Look for any viz_outputs_<airline> directories."""
    return sorted(
        name.replace("viz_outputs_", "")
        for name in os.listdir(BASE)
        if name.startswith("viz_outputs_") and
           os.path.isdir(os.path.join(BASE, name))
    )

app = Flask(__name__)
DATA_PATH  = "convo_metrics.csv"
START_DATE = None
END_DATE   = None

@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")

@app.route("/run", methods=["POST"])
def run_analysis():
    global START_DATE, END_DATE
    START_DATE = request.form["start_date"]
    END_DATE   = request.form["end_date"]

    # 1 mine & extract
    subprocess.run(
        [sys.executable, "influencers.py", START_DATE, END_DATE],
        check=True
    )
    # 2 build all viz
    subprocess.run(
        [sys.executable, "influencers_VIS.py"],
        check=True
    )

    return redirect(url_for("choose_airline"))

@app.route("/airlines", methods=["GET"])
def choose_airline():
    airlines = list_airlines()
    if not airlines:
        return "<p>No airlines found. Run analysis first.</p>"
    return render_template("airline_list.html", airlines=airlines)

@app.route("/dashboard/<airline>", methods=["GET"])
def dashboard(airline):
    dir_name = f"viz_outputs_{airline}"
    full_dir = os.path.join(BASE, dir_name)
    if not os.path.isdir(full_dir):
        abort(404)
    plot_files = sorted(f for f in os.listdir(full_dir) if f.endswith(".html"))
    return render_template(
        "index.html",
        airline=airline,
        plot_files=plot_files
    )

@app.route("/plot/<airline>/<filename>", methods=["GET"])
def show_plot(airline, filename):
    return render_template(
        "plot_view.html",
        airline=airline,
        filename=filename
    )

@app.route("/viz_outputs/<airline>/<path:filename>")
def viz_outputs(airline, filename):
    folder = os.path.join(BASE, f"viz_outputs_{airline}")
    if not os.path.isdir(folder):
        abort(404)
    return send_from_directory(folder, filename)

if __name__ == "__main__":
    app.run(debug=True)
