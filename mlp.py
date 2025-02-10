import torch

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features, requires_grad=False),
            b1 = torch.randn(linear_1_out_features, requires_grad=False),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features, requires_grad=False),
            b2 = torch.randn(linear_2_out_features, requires_grad=False),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        self.cache = dict()

    def activation(self, z, func):
        if func == "relu":
            return torch.relu(z)
        elif func == "sigmoid":
            return torch.sigmoid(z)
        elif func == "identity":
            return z
        else:
            raise ValueError("Unsupported activation function")

    def activation_derivative(self, z, func):
        if func == "relu":
            grad = (z > 0).float()
            grad[z == 0] = 0.5  # Handle ReLU's undefined derivative at 0
            return grad
        elif func == "sigmoid":
            sig = torch.sigmoid(z)
            return sig * (1 - sig)
        elif func == "identity":
            return torch.ones_like(z)
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, x):
        W1, b1 = self.parameters["W1"], self.parameters["b1"]
        W2, b2 = self.parameters["W2"], self.parameters["b2"]

        z1 = x @ W1.T + b1
        a1 = self.activation(z1, self.f_function)
        z2 = a1 @ W2.T + b2
        y_hat = self.activation(z2, self.g_function)

        self.cache = {"x": x, "z1": z1, "a1": a1, "z2": z2, "y_hat": y_hat}

        return y_hat

    def backward(self, dJdy_hat):
        W1, W2 = self.parameters["W1"], self.parameters["W2"]
        x, z1, a1, z2, y_hat = self.cache["x"], self.cache["z1"], self.cache["a1"], self.cache["z2"], self.cache["y_hat"]

        batch_size = x.shape[0]

        dJdz2 = 2 * dJdy_hat * self.activation_derivative(z2, self.g_function)

        dJdW2 = (dJdz2.T @ a1) / batch_size
        dJdb2 = dJdz2.sum(dim=0) / batch_size

        dJda1 = dJdz2 @ W2
        dJdz1 = dJda1 * self.activation_derivative(z1, self.f_function)

        dJdW1 = (dJdz1.T @ x) / batch_size
        dJdb1 = dJdz1.sum(dim=0) / batch_size

        self.grads["dJdW1"] = dJdW1
        self.grads["dJdb1"] = dJdb1
        self.grads["dJdW2"] = dJdW2
        self.grads["dJdb2"] = dJdb2

    def clear_grad_and_cache(self):
        for key in self.grads:
            self.grads[key].zero_()
        self.cache.clear()

def mse_loss(y, y_hat):
    J = torch.mean((y_hat - y) ** 2)
    dJdy_hat = 2 * (y_hat - y) / y.shape[0]
    return J, dJdy_hat

def bce_loss(y, y_hat):
    epsilon = 1e-8
    y_hat = torch.clamp(y_hat, epsilon, 1 - epsilon)
    J = -torch.mean(y * torch.log(y_hat) + (1 - y) * torch.log(1 - y_hat))
    dJdy_hat = (y_hat - y) / (y_hat * (1 - y_hat) * y.shape[0])
    return J, dJdy_hat
