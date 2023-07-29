<h1> ACTIVATION FUNCTION FOR THE MACHINE LEARNING</h1>

Sigmoid:
Formula: f(x) = 1 / (1 + exp(-x))
Description: Sigmoid activation function maps the input to a range between 0 and 1, often used in binary classification problems.

Hyperbolic Tangent (Tanh):
Formula: f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
Description: Tanh activation function maps the input to a range between -1 and 1, providing a centered, zero-mean output.

ReLU (Rectified Linear Unit):
Formula: f(x) = max(0, x)
Description: ReLU is widely used, and it replaces negative values with zero, keeping positive values unchanged.

Leaky ReLU:
Formula: f(x) = max(alpha * x, x), where alpha is a small positive constant.
Description: Leaky ReLU addresses the "dying ReLU" problem by allowing a small negative slope for negative values.

PReLU (Parametric ReLU):
Formula: f(x) = max(alpha * x, x), where alpha is a learnable parameter.
Description: PReLU extends Leaky ReLU by allowing the negative slope to be learned during training.

ELU (Exponential Linear Unit):
Formula: f(x) = x if x >= 0, f(x) = alpha * (exp(x) - 1) if x < 0, where alpha is a small positive constant.
Description: ELU introduces a smooth curve for negative values, helping with the vanishing gradient problem.

SELU (Scaled Exponential Linear Unit):
Formula: f(x) = lambda * x if x >= 0, f(x) = lambda * alpha * (exp(x) - 1) if x < 0, where lambda and alpha are fixed constants.
Description: SELU promotes self-normalization in deep neural networks, making it suitable for deep architectures.

Softmax:
Formula: softmax(z_i) = exp(z_i) / sum(exp(z_j)), for all i in 1 to N.
Description: Softmax is used in multiclass classification problems to convert logits into a probability distribution.
