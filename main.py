import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping


np.random.seed(42)

# Leer el archivo y manejar valores nulos
def read_data(file):
    data = pd.read_csv(file)
    data = data.drop(columns=['Cabin'])
    print("PRE: Valores Nulos detectados en df:")
    print("Shape PRE", data.shape)
    print(data.isna().sum())
    print("POST: Valores Nulos detectados en df:")
    print(data.isna().sum())
    print("Shape POST", data.shape)
    return data

# Mapear variable categórica 'Sex' a valores numéricos
def map_sex(data):
    sex_mapping = {'male': 0, 'female': 1}  
    data['SexNumer'] = data['Sex'].map(sex_mapping)
    return data

# Realizar codificación one-hot de la variable 'Embarked'
def one_hot_encoding(data):
    encoder = OneHotEncoder(sparse=False)
    embarked_encoded = encoder.fit_transform(data[['Embarked']])
    embarked_categories = encoder.categories_[0]
    embarked_df = pd.DataFrame(embarked_encoded, columns=[f'Embarked_{category}' for category in embarked_categories], index=data.index)
    data = pd.concat([data, embarked_df], axis=1)
    return data

# Extraer valores numéricos del boleto y crear una nueva columna 'TicketNum'
def extract_ticket_number(data):
    ticket_num = data['Ticket'].apply(lambda x: re.findall(r'\d+', x))
    ticket_num = ticket_num.apply(lambda x: int(x[0]) if len(x) > 0 else None)
    data['TicketNum'] = ticket_num
    return data

early_stopping = EarlyStopping(monitor='val_loss', patience=8)

def train_evaluate_neural_network(X_train_scaled, y_train, X_test_scaled, y_test):
    model_nn = Sequential()
    model_nn.add(Dense(128, input_shape=(X_train_scaled.shape[1],), activation="relu"))
    model_nn.add(Dense(64, activation="relu"))
    model_nn.add(Dense(32, activation="relu"))
    model_nn.add(Dense(1, activation="sigmoid"))
    model_nn.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    model_nn.summary()
    history = model_nn.fit(X_train_scaled, y_train, epochs=30, validation_data=(X_test_scaled, y_test), callbacks=[early_stopping])
    
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.ylabel('Accuracy')    
    plt.xlabel('Epoch')    
    plt.legend(['Train', 'Test'])
    plt.savefig('Accuracy_epoch.jpg')
    plt.show()

    y_pred_nn_prob = model_nn.predict(X_test_scaled).ravel()
    y_pred_nn = np.round(y_pred_nn_prob).astype(int)
    accuracy_nn = accuracy_score(y_test, y_pred_nn)
    return model_nn, accuracy_nn, confusion_matrix(y_test, y_pred_nn), classification_report(y_test, y_pred_nn)

def main():
    file = 'titanic.csv'

    data = read_data(file)
    print("SHAPE:-------",len(data))
    data = map_sex(data)
    data = one_hot_encoding(data)
    data = extract_ticket_number(data)
    selected_features = data.select_dtypes(include=['float64', 'int64'])

    print("DataFrame con codificación one-hot de 'Embarked' y columna 'TicketNum':")
    print(data.head(10))

    correlation_matrix_selected = selected_features.corr()
   
    plt.figure(figsize=(10, 12))
    heatmap = sns.heatmap(correlation_matrix_selected, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=30, fontname='Arial', fontsize=10)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontname='Arial', fontsize=10, rotation=30)
    plt.title('Matriz de Correlación (Means)')
    plt.savefig('Matriz de Correlación (Means).jpg')
    plt.show()
   
    X = selected_features.drop(columns=['Survived'])
    y = selected_features['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    model_nn, accuracy_nn, conf_matrix_nn, report_nn = train_evaluate_neural_network(X_train_scaled, y_train, X_test_scaled, y_test)
    print('Precisión de la Red Neuronal:', accuracy_nn)
    print('Matriz de Confusión de la Red Neuronal:\n', conf_matrix_nn)
    print('Informe de Clasificación de la Red Neuronal:\n', report_nn)

    y_pred_nn_prob = model_nn.predict(X_test_scaled).ravel()
    fpr_nn, tpr_nn, _ = roc_curve(y_test, y_pred_nn_prob)
    auc_nn = roc_auc_score(y_test, y_pred_nn_prob)

    plt.figure(figsize=(8, 6))

    plt.plot(fpr_nn, tpr_nn, label=f'Neural Network (AUC = {auc_nn:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig('ROC Curve.jpg')
    plt.show()
    print("SHAPE:-------",len(data))

if __name__ == "__main__":
    main()
