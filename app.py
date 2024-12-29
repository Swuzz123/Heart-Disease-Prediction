from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load mô hình đã huấn luyện
with open("model/heart_disease_model.pickle", "rb") as f:
    model = pickle.load(f)
    
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods= ['POST'])
def predict():
    try:
        # Lấy dữ liệu từ form HTML
        data = request.form
        age = int(data['age'])
        resting_bp = int(data['resting_bp'])
        cholesterol = int(data['cholesterol'])
        fasting_bs = int(data['fasting_bs'])
        resting_ecg = int(data['resting_ecg'])
        max_hr = int(data['max_hr'])
        oldpeak = float(data['oldpeak'])
        st_slope = int(data['st_slope'])
        sex = data['sex']
        chest_pain_type = data['chest_pain_type']
        exercise_angina = data['exercise_angina']

        # Chuẩn bị dữ liệu để dự đoán
        x = np.zeros(len(model.feature_names_in_))
        x[0] = age
        x[1] = resting_bp
        x[2] = cholesterol
        x[3] = fasting_bs
        x[4] = resting_ecg
        x[5] = max_hr
        x[6] = oldpeak
        x[7] = st_slope
        
        # Mã hóa categorical features
        if sex == "F":
            x[np.where(model.feature_names_in_ == 'Sex_F')[0][0]] = 1
        elif sex == "M":
            x[np.where(model.feature_names_in_ == 'Sex_M')[0][0]] = 1

        if chest_pain_type == "ASY":
            x[np.where(model.feature_names_in_ == 'ChestPainType_ASY')[0][0]] = 1
        elif chest_pain_type == "ATA":
            x[np.where(model.feature_names_in_ == 'ChestPainType_ATA')[0][0]] = 1
        elif chest_pain_type == "NAP":
            x[np.where(model.feature_names_in_ == 'ChestPainType_NAP')[0][0]] = 1
        elif chest_pain_type == "TA":
            x[np.where(model.feature_names_in_ == 'ChestPainType_TA')[0][0]] = 1

        if exercise_angina == "N":
            x[np.where(model.feature_names_in_ == 'ExerciseAngina_N')[0][0]] = 1
        elif exercise_angina == "Y":
            x[np.where(model.feature_names_in_ == 'ExerciseAngina_Y')[0][0]] = 1

        # Dự đoán
        prediction = model.predict([x])[0]
        result = "Has Disease" if prediction == 1 else "No Disease"

        return render_template('index.html', prediction_text=f"Prediction: {result}")
    
    except Exception as e:
        return jsonify({"error": str(e)})
    
if __name__ == "__main__":
    app.run(debug= True)