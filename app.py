from flask import Flask, render_template, request, jsonify
from src.pipeline.preediction_pipeline import PredictionPipeline, CustomDataClass  # Fixed: preediction_pipeline

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def prediction_data():
    if request.method == "GET":
        return render_template("home.html")
    
    else:
        try:
            # Input validation + prediction
            data = CustomDataClass(
                age=int(request.form.get("age")),
                workclass=int(request.form.get("workclass")),
                education_num=int(request.form.get("education_num")),
                marital_status=int(request.form.get("marital_status")),
                occupation=int(request.form.get("occupation")),
                relationship=int(request.form.get("relationship")),
                race=int(request.form.get("race")),
                sex=int(request.form.get("sex")),
                capital_gain=int(request.form.get("capital_gain")),
                capital_loss=int(request.form.get("capital_loss")),
                hours_per_week=int(request.form.get("hours_per_week")),
                native_country=int(request.form.get("native_country"))
            )

            final_data = data.get_data_as_dataframe()
            pipeline_prediction = PredictionPipeline()
            pred = pipeline_prediction.predict(final_data)[0]  # Fixed: get scalar value

            # Clean result formatting
            result_text = "Your Yearly Income is More than 50k" if pred == 1 else "Your Yearly Income is Less than Equal to 50k"
            
            return render_template("results.html", final_result=result_text, prediction=pred)
            
        except Exception as e:
            return render_template("results.html", final_result=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=8080)  # Added port
