from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the full pipeline (preprocessing + model)
with open('model.pickle', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')  # Your HTML form file

@app.route('/predict', methods=['POST'])
def predict():
    try:
        sqft = float(request.form['sqft'])
        bhk = int(request.form['uiBHK'])
        bath = int(request.form['uiBathrooms'])
        location = request.form['location']

        input_df = pd.DataFrame([[location, sqft, bath, bhk]],
                                columns=['location', 'total_sqft', 'bath', 'bhk'])

        prediction = model.predict(input_df)[0]
        output = round(prediction, 2)
        return render_template('index.html', prediction_text=f"Estimated Price: ₹{output} Lakhs")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
