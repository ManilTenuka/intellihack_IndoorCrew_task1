from utils.data_utils import load_data, preprocess_data, split_data
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from collections import Counter


def main():
    file_path = 'data/Crop_Dataset.csv'
    data = load_data(file_path)

    # The 'Label' column should be stored before it's transformed or dropped.
    unique_labels = np.unique(data['Label'])

    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Model Training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Model Evaluation
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'Model Accuracy: {accuracy:.2f}')

    # Find the top three most frequently predicted crop types
    predicted_labels = [unique_labels[pred] for pred in predictions]
    top_three_crops = [item for item, count in Counter(predicted_labels).most_common(3)]

    print("Top Three Recommended Crops:", top_three_crops)


if __name__ == "__main__":
    main()
