from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)
label_encoder = pickle.load(open('api/models/label_encoder.pkl', 'rb'))
models = {'svm_poly': None, 'svm_rbf': None, 'dt': None, 'knn': None, 'gnb': None, 'rf': None}

def loadModels():
    for model in models:
        models[model] = pickle.load(open('api/models/'+model+'.pkl', 'rb'))



@app.route('/', methods=['GET'])
def predict_priority():
    args = request.args.to_dict()
    print(args)
    parameters = ['age', 'income', 'disabled', 'pregnant', 'bmi', 'doctor_severity_score', 'disease_index', 'deadline', 'claiming_amount', 'sex', 'successful_previous_claims']
    input = np.array([float(args[parameter]) for parameter in parameters])
    output = {model:label_encoder.inverse_transform(models[model].predict(input.reshape(1, -1))).flat[0] for model in models}
    print(output)
    return jsonify(output)

if __name__ == '__main__': 
    loadModels()  
    app.run(debug=True)