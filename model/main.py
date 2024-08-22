import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
from dotenv import load_dotenv


def get_processed_data():
    load_dotenv()
    data = pd.read_csv(os.getenv('DATA_FILE'))
    data.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
    data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
    return data

def create_model(data):
    if data is None:
        return None, None
    
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']

    scaler = StandardScaler()
    X = scaler.fit_transform(X.values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f'Accuracy of our model: {accuracy_score(y_test, y_pred)}')
    print(f"Classification report: \n {classification_report(y_test, y_pred)}")
    return model, scaler

def get_model():
    load_dotenv()

    data = get_processed_data()

    model, scaler = create_model(data)

    if model is not None and scaler is not None:

        with open(os.getenv('MODEL_DIR') + os.getenv('MODEL_FILE'), 'wb') as file:
            pickle.dump(model, file)

        with open(os.getenv('MODEL_DIR') + os.getenv('SCALER_FILE'), 'wb') as file:
            pickle.dump(scaler, file)

if __name__ == '__main__':
    get_model()