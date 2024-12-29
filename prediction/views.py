from django.shortcuts import render
from django.http import JsonResponse
import pickle
import joblib
import pandas as pd

def home(request):
    return render(request, 'home.html')


def modelPrediction(data, model):
    # Load the model from file
    with open(f'prediction/model/{model}_model.pkl', 'rb') as f:
        fit_model = joblib.load(f)

    # Pastikan input data tidak memiliki feature names
    data = data.values
    result = fit_model.predict(data)
    print("hasil:", result)

    return result

def confidence(data, model):
    # Load the model from file
    with open(f'prediction/model/{model}_model.pkl', 'rb') as f:
        fit_model = joblib.load(f)

    # Pastikan input data tidak memiliki feature names
    data = data.values
    result = fit_model.predict_proba(data)
    print("confidence:", result)

    return result

def predict(request):
    if request.method == "POST":
        # Ambil data dari form
        age = request.POST.get('age')
        gender = request.POST.get('gender')
        ticket_class = request.POST.get('class')
        fare = request.POST.get('tarif')
        cabin = request.POST.get('cabin')
        model_req = request.POST.get('model')

        # Buat DataFrame
        data = pd.DataFrame({
            'Age': [age],
            'Sex': [gender],
            'Passenger Class': [ticket_class],
            'Passenger Fare': [fare],
            'Cabin': [cabin],
        })

        # Pastikan data prediksi sesuai urutan kolom pelatihan
        data = data.reindex(columns=['Passenger Class', 'Sex', 'Age', 'Passenger Fare', 'Cabin'])

        # Prediksi menggunakan model
        if model_req == "1":
            model = "knn"
        elif model_req == "2":
            model = "logistic_regression"
        elif model_req == "3":
            model = "naive_bayes"
        elif model_req == "4":
            model = "svm"
        else:
            model = "knn"

        try:
            result = modelPrediction(data, model)
            probabilitas = confidence(data, model)
            survived_con = probabilitas[0][1] * 100
            not_survived_con = probabilitas[0][0] * 100
            # print("confidence:", confidence_score)
            print("model:", model)
            prediction = "Survived" if result[0] == 1 else "Not Survived"
            return JsonResponse({
                'prediction': prediction,
                'survived_con': survived_con,
                'not_survived_con': not_survived_con,
                })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request'}, status=400)