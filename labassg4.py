import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n_samples = 500
X_base = np.random.rand(n_samples, 1)
X = np.hstack([X_base + np.random.normal(0, 0.02, (n_samples, 1)) for _ in range(7)])
y = 3 * X_base.flatten() + np.random.normal(0, 0.1, n_samples)

data = pd.DataFrame(X, columns=[f'Feature_{i+1}' for i in range(X.shape[1])])
data['Target'] = y

X = data.drop(columns='Target').values
y = data['Target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def ridge_regression_gradient_descent(X, y, alpha, lr, epochs):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0
    cost_history = []

    for epoch in range(epochs):
        y_pred = np.dot(X, weights) + bias
        residuals = y_pred - y
        dw = (2 / n_samples) * np.dot(X.T, residuals) + 2 * alpha * weights
        db = (2 / n_samples) * np.sum(residuals)
        weights -= lr * dw
        bias -= lr * db
        cost = (1 / n_samples) * np.sum(residuals**2) + alpha * np.sum(weights**2)
        cost_history.append(cost)
    
    return weights, bias, cost_history

learning_rates = [0.0001, 0.001, 0.01, 0.1, 1, 10]
alphas = [1e-15, 1e-10, 1e-5, 1e-3, 0, 1, 10, 20]
epochs = 1000

best_params = {}
best_r2 = float('-inf')
best_cost = float('inf')

for lr in learning_rates:
    for alpha in alphas:
        weights, bias, cost_history = ridge_regression_gradient_descent(X_train, y_train, alpha, lr, epochs)
        y_pred = np.dot(X_test, weights) + bias
        cost = mean_squared_error(y_test, y_pred) + alpha * np.sum(weights**2)
        r2 = r2_score(y_test, y_pred)
        if r2 > best_r2 or (r2 == best_r2 and cost < best_cost):
            best_r2 = r2
            best_cost = cost
            best_params = {'learning_rate': lr, 'alpha': alpha, 'weights': weights, 'bias': bias}

print("Best Parameters:")
print(f"  Learning Rate: {best_params['learning_rate']}")
print(f"  Regularization Parameter (alpha): {best_params['alpha']}")
print(f"  Minimum Cost: {best_cost}")
print(f"  Maximum R^2 Score: {best_r2}")

final_weights = best_params['weights']
final_bias = best_params['bias']
final_y_pred = np.dot(X_test, final_weights) + final_bias

print("\nFinal Model Performance:")
print(f"Mean Squared Error: {mean_squared_error(y_test, final_y_pred):.4f}")
print(f"R^2 Score: {r2_score(y_test, final_y_pred):.4f}")