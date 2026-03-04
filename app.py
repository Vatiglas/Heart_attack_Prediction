import streamlit as st
import joblib
import pandas as pd

#load pkl nya
scaler = joblib.load("artifacts/preprocessor.pkl")
model = joblib.load("artifacts/model.pkl")

def main():
    st.title('Heart Attack Prediction App')

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input('Umur (age)', min_value=1, max_value=120, value=50)
        sex = st.selectbox('Jenis Kelamin (sex)', options=[0, 1], format_func=lambda x: "1 (Pria)" if x == 1 else "0 (Wanita)")
        cp = st.selectbox('Tipe Nyeri Dada (cp)', options=[0, 1, 2, 3])
        trestbps = st.number_input('Tekanan Darah Istirahat (trestbps)', min_value=50, max_value=250, value=120)
        chol = st.number_input('Kolesterol (chol)', min_value=100, max_value=600, value=200)
        fbs = st.selectbox('Gula Darah Puasa > 120 mg/dl (fbs)', options=[0, 1])
        restecg = st.selectbox('Hasil EKG Istirahat (restecg)', options=[0, 1, 2])

    with col2:
        thalach = st.number_input('Detak Jantung Maksimal (thalach)', min_value=60, max_value=250, value=150)
        exang = st.selectbox('Angina karena Olahraga (exang)', options=[0, 1], format_func=lambda x: "1 (Ya)" if x == 1 else "0 (Tidak)")
        oldpeak = st.number_input('Depresi ST (oldpeak)', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox('Kemiringan Segmen ST (slope)', options=[0, 1, 2])
        ca = st.selectbox('Jumlah Pembuluh Darah Utama (ca)', options=[0, 1, 2, 3, 4])
        thal = st.selectbox('Thalassemia (thal)', options=[0, 1, 2, 3])
    
    
    # 3. Tombol Prediksi
    if st.button('Prediksi Risiko', use_container_width=True):
        features = {
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal
        }
        
        result = make_prediction(features)
        
        # 4. Menampilkan Hasil (Asumsi: 1 = Risiko Tinggi, 0 = Risiko Rendah)
        if result == 1:
            st.error('Hasil Prediksi: Pasien Berisiko Tinggi Terkena Serangan Jantung')
        else:
            st.success('Hasil Prediksi: Pasien Berisiko Rendah Terkena Serangan Jantung')

def make_prediction(features):
    input_df = pd.DataFrame([features])
    cols_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])
    prediction = model.predict(input_df)
    return prediction[0]

if __name__ == '__main__':
    main()