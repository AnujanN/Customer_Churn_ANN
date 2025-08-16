# Customer Churn Prediction with Artificial Neural Network

A machine learning project that predicts customer churn using an Artificial Neural Network (ANN) built with TensorFlow/Keras and deployed using Streamlit.

## 🚀 Features

- **Data Preprocessing**: Automated encoding of categorical variables and feature scaling
- **Deep Learning Model**: Custom ANN architecture for binary classification
- **Interactive Web App**: User-friendly Streamlit interface for real-time predictions
- **Model Persistence**: Saved encoders and scalers for consistent predictions
- **TensorBoard Integration**: Training visualization and monitoring

## 📋 Requirements

- Python 3.8+
- TensorFlow/Keras
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Pickle

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/customer_churn_ANN.git
cd customer_churn_ANN
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy environment file:
```bash
cp .env.example .env
```

## 📊 Dataset

The project uses the `Churn_Modelling.csv` dataset containing customer information including:
- Demographics (Age, Gender, Geography)
- Account details (Balance, Tenure, Number of Products)
- Behavior (Credit Card ownership, Active membership)
- Target variable (Exited - whether customer churned)

## 🔧 Usage

### Training the Model

1. Open and run the Jupyter notebook:
```bash
jupyter notebook experiments.ipynb
```

2. Execute all cells to:
   - Load and preprocess data
   - Train the ANN model
   - Save the trained model and encoders

### Running the Web Application

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

### Using TensorBoard

To view training metrics:
```bash
tensorboard --logdir logs/fit
```

## 🏗️ Model Architecture

- **Input Layer**: 12 features (after preprocessing)
- **Hidden Layer 1**: 64 neurons with ReLU activation
- **Hidden Layer 2**: 32 neurons with ReLU activation
- **Output Layer**: 1 neuron with Sigmoid activation
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy

## 📁 Project Structure

```
customer_churn_ANN/
├── app.py                           # Streamlit web application
├── experiments.ipynb                # Model training notebook
├── Churn_Modelling.csv             # Dataset
├── customer_churn_model.h5         # Trained model
├── label_encoder_gender.pkl        # Gender encoder
├── onehot_encoder_geography.pkl    # Geography encoder
├── scaler.pkl                      # Feature scaler
├── requirements.txt                # Dependencies
├── .gitignore                      # Git ignore file
├── .env.example                    # Environment variables template
└── README.md                       # Project documentation
```

## 🎯 Model Performance

The model achieves:
- Training Accuracy: ~86%
- Validation Accuracy: ~85%
- Early stopping implemented to prevent overfitting

## 🌐 Deployment

The Streamlit app can be deployed on various platforms:
- Streamlit Cloud
- Heroku
- AWS EC2
- Google Cloud Platform

## 👨‍💻 Developer

**Anujan**  
Faculty of IT, University of Moratuwa

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

⭐ If you found this project helpful, please give it a star!
