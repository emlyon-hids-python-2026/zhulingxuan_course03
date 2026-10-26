"""Hello world, then train a simple machine learning model on the iris dataset."""

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

print("Hello, world!")
print("Welcome to the Iris Project Course 03 of the HIDS Python Programming course!")
print("This program was made by lingxuan.zhu@edu.em-lyon.com")

# %% Load the data
X, y = load_iris(return_X_y=True, as_frame=True)
print(f"Dataset: {X.shape[0]} flowers, {X.shape[1]} features")

# %% Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# %% Train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# %% Evaluate the model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred, target_names=load_iris().target_names))
