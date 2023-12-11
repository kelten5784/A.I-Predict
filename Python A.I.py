from sklearn.linear_model import LinearRegression
import numpy as np

class NumberPredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.X = np.array([])
        self.y = np.array([])

    def train(self, user_input, actual_number):
        self.X = np.append(self.X, user_input)
        self.y = np.append(self.y, actual_number)
        X_reshaped = self.X.reshape(-1, 1)
        self.model.fit(X_reshaped, self.y)

    def predict(self, user_input):
        input_reshaped = np.array(user_input).reshape(-1, 1)
        predicted_number = self.model.predict(input_reshaped)
        return np.clip(predicted_number, 1, 100)[0]

if __name__ == "__main__":
    predictor = NumberPredictor()

    # Generate 10 random numbers and make predictions
    for _ in range(10):
        actual_number = np.random.uniform(1, 100)
        predictor.train(actual_number, actual_number)
        predicted_number = predictor.predict(actual_number)

        print(f"Actual: {actual_number:.2f}, Predicted: {predicted_number:.2f}")
