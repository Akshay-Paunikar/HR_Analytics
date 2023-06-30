import numpy as np
import pandas as pd
from flask import Flask, render_template, request

from hr_analytics.pipeline.prediction_pipeline import CustomData, PredictPipeline
application = Flask(__name__)

app = application

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    
    else:
        data = CustomData(
            department = request.form.get('department'),
            region = request.form.get('region'),
            education = request.form.get('education'),
            gender = request.form.get('gender'),
            recruitment_channel = request.form.get('recruitment_channel'),
            no_of_trainings = float(request.form.get('no_of_trainings')),
            age = float(request.form.get('age')),
            previous_year_rating = float(request.form.get('previous_year_rating')),
            length_of_service = float(request.form.get('length_of_service')),
            KPIs_met = request.form.get('KPIs_met'),
            awards_won = request.form.get('awards_won'),
            avg_training_score = float(request.form.get('avg_training_score'))                                    
        )
        
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        if results == 0:
            promoted = "No"
        else:
            promoted = "Yes"
        return render_template('index.html', results=promoted)
    
if __name__ == "__main__":
    app.run(debug=True)