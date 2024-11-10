from model import Model
from flask import Flask, render_template, request
import re

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

    type = form.get('type').upper()
    borough = form.get('borough')
    zip = form.get('zip')

    # ERRORS

    # Types
    allowed_values_type = {
        'INJURY', 'SICK', 'PEDSTR', 'CARD', 'DRUG', 'MVAINJ', 'ASTHMB', 'OTHER', 'INJMAJ', 
        'UNKNOW', 'UNC', 'STAB', 'MCI21', 'PD13C', 'EDPM', 'CARDBR', 'DIFFBR', 'OBLAB', 
        'SICMIN', 'EDP', 'INBLED', 'COLD', 'RESPIR', 'ELECT', 'EDPC', 'ABDPN', 'STATEP', 
        'SICPED', 'BURNMA', 'OBMAJ', 'SHOT', 'ALTMEN', 'BURNMI', 'TRAUMA', 'CVAC', 'MEDRXN', 
        'ANAPH', 'ARREST', 'STNDBY', 'T-TEXT', 'T-UNC', 'OBCOMP', 'CVA', 'PD13', 'CHOKE', 
        'INJMIN', 'SEIZR', 'AMPMAJ', 'DIFFFC', 'CDBRFC', 'MVA', 'INHALE'
    }

    if type not in allowed_values_type:
        raise ValueError(f"Invalid incident type: {type}. Allowed values are: {', '.join(allowed_values_type)}")

    # Boroughs
    allowed_boroughs = {
        'QUEENS', 'MANHATTAN', 'BROOKLYN', 'BRONX', 'RICHMOND / STATEN ISLAND'
    }

    if borough not in allowed_boroughs:
        raise ValueError(f"Invalid borough: {borough}. Allowed values are: {', '.join(allowed_boroughs)}")
    
    
    # Zip codes
    pattern_zip = r'^\d{5}$'
    if not re.match(pattern_zip, zip):
        raise ValueError(f"Invalid zip: {zip}. Zip code must be non-epty string that consists of only 5 digits")


    # MODEL

    result = model.predict(type, borough, zip)
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
    app.run(debug=True, port=5001)


