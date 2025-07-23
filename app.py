from flask import Flask, render_template, request, Response
import io
import sys
import time
from flask import jsonify

import automated as at

application = Flask(__name__)

def generate_bytes(generator):
    for chunk in generator:
        yield chunk.encode('utf-8') 

@application.route("/")
def index():
    return render_template("index.html")

@application.route("/get_init_species", methods=["POST"])
def get_init_species_route():
    file1 = request.files.get("file1")
    if not file1:
        return jsonify({"error": "Missing file"}), 400

    try:
        species_list = at.get_init_species(file1)  # call existing function
        # species_list = ["SM", "base", "reagent"]
        return jsonify({"species": species_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    
@application.route("/run", methods=["POST"])
def run():
    file1 = request.files.get("file1")

    if not file1:
        return "Missing file", 400

    try:
        final_timepoint = float(request.form.get(f"Final"))
        interval = float(request.form.get(f"Interval"))
        k1 = float(request.form.get(f"k1"))
        E1 = float(request.form.get(f"E1"))
        k2 = float(request.form.get(f"k2"))
        E2 = float(request.form.get(f"E2"))
        tolerance = float(request.form.get(f"imptol"))
        tolerance /= 100
        file_data = file1.read()
        file_buffer = io.BytesIO(file_data)
        names = request.form.getlist("species_name[]")
        values = request.form.getlist("species_value[]")

        # Convert string values to floats
        values = [float(v) for v in values]

        # Zip them into a dictionary
        species_dict = dict(zip(names, values))

        print("Species Dictionary:", species_dict)
        templow = float(request.form.get(f"templow"))
        temphigh = float(request.form.get(f"temphigh"))
    except ValueError:
        return "Weights nonnumeric or file unreadable", 400

    return Response(generate_bytes(at.run_automated(file_buffer, [0.1,0,0,0.9,0.1], final_timepoint, interval, k1, k2, E1, E2, species_dict, templow, temphigh, tolerance)), mimetype="text/plain")

if __name__ == "__main__":
    application.run(host='0.0.0.0', port=5000,debug=True)
