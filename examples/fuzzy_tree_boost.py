import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from fuzzytree._classes import FuzzyGBDT


# Step 1: Generate synthetic data for regression
def generate_data(n_samples=1000, noise=0.1):
    """Generate synthetic data for testing."""
    X = np.random.rand(n_samples, 5) * 10  # Features in range [0, 10]
    y = (
        2 * np.sin(X[:, 0])
        + 0.5 * X[:, 1] ** 2
        - X[:, 2]
        + np.random.normal(scale=noise, size=n_samples)
    )  # Target with some noise
    return X, y

def test_fuzzy_regression_tree_boosting():
    # Step 2: Prepare data
    X, y = generate_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Initialize and train the FuzzyGradientBoostingRegressor
    model = FuzzyGBDT(n_estimators=10, learning_rate=0.1)

    print("Training Fuzzy Gradient Boosting Regressor...")
    model.fit(X_train, y_train)

    # Step 4: Evaluate the model
    print("Evaluating the model...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Mean Squared Error on Test Data: {mse:.4f}")

    import matplotlib.pyplot as plt

    # Step 6: Visualize predictions vs. actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='blue', edgecolors='k', label='Predictions')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal Fit')
    plt.title('Predicted vs. Actual Values')
    plt.xlabel('Actual Values (y_test)')
    plt.ylabel('Predicted Values (y_pred)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    test_fuzzy_regression_tree_boosting()
