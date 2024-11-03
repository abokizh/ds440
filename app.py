from model import Model
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # User Input
    form = request.form
    
    model = Model()
    model.train()

    result = model.predict(form.get('type'), form.get('borough'), form.get('zip'))
    minutes = int(result[0])
    seconds = int((result[0] - minutes) * 60)

    error = model.test_mse**(1/2)
    minutes_error = int(error)
    seconds_error = int((error - minutes_error) * 60)

    prediction = {
        "prediction": [minutes, seconds],
        "error": [minutes_error, seconds_error],
    }

    return render_template('index.html', result=prediction)

if __name__ == '__main__':
    app.run(debug=True)


