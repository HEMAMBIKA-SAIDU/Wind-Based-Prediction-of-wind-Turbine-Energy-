from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Dummy weather data (you can replace with API later)
city_weather = {
    "Hyderabad": {"temperature": 32, "humidity": 45, "pressure": 1012, "wind_speed": 5},
    "Chennai": {"temperature": 34, "humidity": 60, "pressure": 1009, "wind_speed": 6},
    "Mumbai": {"temperature": 30, "humidity": 70, "pressure": 1010, "wind_speed": 4},
    "Delhi": {"temperature": 28, "humidity": 50, "pressure": 1015, "wind_speed": 3},
    "Bangalore": {"temperature": 26, "humidity": 55, "pressure": 1013, "wind_speed": 5},
}

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    weather = None
    prediction = None

    if request.method == "POST":

        # If weather button clicked
        if "get_weather" in request.form:
            city = request.form["city"]
            weather = city_weather.get(city)

        # If prediction button clicked
        if "predict_energy" in request.form:
            theoretical_power = float(request.form["theoretical_power"])
            wind_speed = float(request.form["wind_speed"])

            input_data = np.array([[theoretical_power, wind_speed]])
            prediction = round(model.predict(input_data)[0], 2)

    return render_template("dashboard.html",
                           weather=weather,
                           prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)