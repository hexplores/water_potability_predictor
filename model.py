from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

def train_and_compare_models(df):
    # Prepare features and target
    X = df.iloc[:, :9]
    y = df['Potability']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=40
    )

    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(random_state=40),
        "Support Vector Machine": SVC(probability=True, random_state=40),
        "K-Nearest Neighbors": KNeighborsClassifier()
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        results[name] = {
            "Accuracy": acc,
            "F1 Score": f1,
            "Precision": precision,
            "Recall": recall,
            "Classification Report": classification_report(y_test, y_pred)
        }

    # Print comparison summary
    print("Model Performance Comparison:\n")
    for model_name, metrics in results.items():
        print(f"--- {model_name} ---")
        print(f"Accuracy: {metrics['Accuracy']:.4f}")
        print(f"F1 Score: {metrics['F1 Score']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"Recall: {metrics['Recall']:.4f}")
        print(f"\nClassification Report:\n{metrics['Classification Report']}")
        print("-----------------------------------------------------\n")

    return results
