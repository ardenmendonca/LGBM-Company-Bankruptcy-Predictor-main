from flask import Flask, render_template, request
import pandas as pd
# import joblib
import pickle
import os

app = Flask(__name__)
# path=r'lgbm_best.pkl'
model = pickle.load(open('lgbm_best.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load the ML model and the input data
    # model = joblib.load('model.pkl')
    data = pd.read_csv('gs://examplecsv506/data_for_web.csv')
    # file_path = os.path.join('temp', data)
    # data.save(file_pat

    predictions = model.predict(data)

    # Convert predictions to binary (0 or 1) and map to output sentences
    output_sentences = ['Based on the values provided the company will go bankrupt in the future'  if pred < 0.5 else 'Based on the values provided the company will not go bankrupt in the future' for pred in predictions]

    # os.remove(file_path)
    # Render the output template
    return render_template('output.html', sentences=output_sentences)


if __name__ == "__main__":
    app.run(debug=True)
