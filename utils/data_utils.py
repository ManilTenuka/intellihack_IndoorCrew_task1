import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
 
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print("File not found.")
        raise
    return data

def preprocess_data(data):

    # Impute missing values
    for col in data.columns:
        if data[col].dtype == "object":
            # Use mode for categorical data
            mode_value = data[col].mode()[0]
            data[col] = data[col].fillna(mode_value)
        else:
            # Use mean for numerical data
            mean_value = data[col].mean()
            data[col] = data[col].fillna(mean_value)

    # Encode categorical variables
    label_encoder = LabelEncoder()
    if 'Label' in data.columns:
        data['Label_Encoded'] = label_encoder.fit_transform(data['Label'])
        data.drop('Label', axis=1, inplace=True)
    y = data['Label_Encoded']
    X = data.drop('Label_Encoded', axis=1)

    # Standard scaling features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Split the data into train and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
