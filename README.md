# DA6401 Deep Learning - DA24D402 Shuvrajeet Das

Assignment 1:
The goal of this assignment is twofold:

- Implement and use gradient descent (and its variants) with backpropagation for a classification task
- Get familiar with wandb which is a cool tool for running and keeping track of a large number of experiments

## Python versions (including libraries):

```
Python - 3.11.6
Numpy - 1.26.3
Matplotlib - 3.8.2
Wandb - 0.16.2
Tensorflow - 2.18.0
```

or you can use the conda setp yaml as follows
```bash
conda env create -f env_setup.yml
```

## Setup:

```bash
python3 -m pip install numpy
python3 -m pip install matplotlib
python3 -m pip install wandb
python3 -m pip install tensorflow[and-cuda]
```

## Code organisation:
```bash
DL_Assignment
├── activation.py
├── details.txt
├── learning_annealers.py
├── LICENSE
├── losses.py
├── main.py
├── neural_network.py
├── optimizers.py
├── __pycache__
├── README.md
├── run.sh
├── train.py
├── train_used.py
├── utils.py
└── wandb
```

## Project Structure:

The project contains the implementtaion of various algorithms for running a Deep Neural Network.

## Running Procedure:

```
python train.py --wandb_entity myname --wandb_project myprojectname
```

### Arguments to be supported

| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-wp`, `--wandb_project` | myprojectname | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | myname  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-d`, `--dataset` | fashion_mnist | choices:  ["mnist", "fashion_mnist"] |
| `-e`, `--epochs` | 1 |  Number of epochs to train neural network.|
| `-b`, `--batch_size` | 4 | Batch size used to train neural network. | 
| `-l`, `--loss` | cross_entropy | choices:  ["mean_squared_error", "cross_entropy"] |
| `-o`, `--optimizer` | sgd | choices:  ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"] | 
| `-lr`, `--learning_rate` | 0.1 | Learning rate used to optimize model parameters | 
| `-m`, `--momentum` | 0.5 | Momentum used by momentum and nag optimizers. |
| `-beta`, `--beta` | 0.5 | Beta used by rmsprop optimizer | 
| `-beta1`, `--beta1` | 0.5 | Beta1 used by adam and nadam optimizers. | 
| `-beta2`, `--beta2` | 0.5 | Beta2 used by adam and nadam optimizers. |
| `-eps`, `--epsilon` | 0.000001 | Epsilon used by optimizers. |
| `-w_d`, `--weight_decay` | .0 | Weight decay used by optimizers. |
| `-w_i`, `--weight_init` | random | choices:  ["random", "Xavier"] | 
| `-nhl`, `--num_layers` | 1 | Number of hidden layers used in feedforward neural network. | 
| `-sz`, `--hidden_size` | 4 | Number of hidden neurons in a feedforward layer. |
| `-a`, `--activation` | sigmoid | choices:  ["identity", "sigmoid", "tanh", "ReLU"] |

## Runs for generating the assignment

```
# for sweep generation
python train.py --sweep 

# for cross entropy vs squared loss
python3 train.py -e 25 -o adam -lr 0.0005 -w_d 0.0005 -nhl 3 -sz 128 -a tanh -loss cross_entropy -w_i xavier_normal
python3 train.py -e 25 -o adam -lr 0.0005 -w_d 0.0005 -nhl 3 -sz 128 -a tanh -loss squared_error -w_i xavier_normal

# best 3 config for mnist
python3 train.py -e 25 -o adam -lr 0.0005 -w_d 0.0005 -nhl 3 -sz 128 -a tanh -loss cross_entropy -w_i xavier_normal -b 64 -d mnist
python3 train.py -e 25 -o adam -lr 0.0005 -w_d 0.0005 -nhl 4 -sz 64 -a tanh -loss cross_entropy -w_i xavier_normal -b 128 -d mnist
python3 train.py -e 25 -o nadam -lr 0.0005 -w_d 0.0005 -nhl 3 -sz 64 -a tanh -loss cross_entropy -w_i xavier_normal -b 64 -d mnist
```


## Features:

### Activation Functions:

- [x] relu (Rectified Linear Unit Activation) $$f(x) = \max(0, x)$$
- [x] sigmoid (Sigmoid Activation) $$f(x) = \frac{1}{1 + e^{-x}} $$
- [x] tanh (Hyperbolic Tangent Activation) $$f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} $$
- [x] selu (Scaled Exponential Linear Unit Activation) $$f(x) = \text{scale} \times (\max(0, x) + \min(0, \alpha \times (\exp(x) - 1))) $$, where $\alpha \approx 1.6733$ and $\text{scale} \approx 1.0507$
- [x] gelu (Gaussian Error Linear Unit Activation) $$f(x) = x \times \Phi(x) $$, where $\Phi(x)$ is the cumulative distribution function of the standard normal distribution
- [x] leaky_relu (Leaky Rectified Linear Unit Activation) $$f(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{otherwise} \end{cases} $$, where $\alpha$ is a small positive constant
- [x] elu (Exponential Linear Unit Activation) $$f(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha \times (e^x - 1) & \text{otherwise} \end{cases} $$, where $\alpha$ is a small positive constant
- [x] swish (Swish Activation) $$f(x) = x \times \sigma(\beta \times x) $$, where $\sigma(x)$ is the sigmoid function and $\beta$ is a constant
- [x] softplus (Softplus Activation) $$f(x) = \ln(1 + e^x) $$
- [x] mish (Mish Activation) $$f(x) = x \times \tanh(\ln(1 + e^x)) $$
- [x] softmax (Softmax Activation) $$f(x*i) = \frac{e^{x_i}}{\sum*{j=1}^{N} e^{x_j}} $$ for $ i = 1, 2, \ldots, N $

### Optimizers:

- [x] sgd (Stochastic Gradient Descent)
      $$ w_{t+1} = w_t - \eta \nabla L(w_t) $$

- [x] momentum*gd (Momentum Gradient Descent)
      $$ v_{t+1} = \gamma v_t + \eta \nabla L(w_t) $$
	    $$ w_{t+1} = w_t - v_{t+1} $$

- [x] nesterov*gd (Nesterov Accelerated Gradient)
      $$ v_{t+1} = \gamma v_t + \eta \nabla L(w_t - \gamma v_t) $$
      $$ w_{t+1} = w_t - v_{t+1} $$

- [x] adagrad (Adaptive Gradient Algorithm)
      $$ G_{t+1} = G_t + (\nabla L(w_t))^2 $$
      $$ w_{t+1} = w_t - \frac{\eta}{\sqrt{G_{t+1}} + \epsilon} \nabla L(w_t) $$

- [x] rmsprop (Root Mean Square Propagation)
      $$ G_{t+1} = \beta G_t + (1 - \beta) (\nabla L(w_t))^2 $$
      $$ w_{t+1} = w_t - \frac{\eta}{\sqrt{G_{t+1}} + \epsilon} \nabla L(w_t) $$

- [x] adadelta (Adaptive Delta)
      $$ G_{t+1} = \beta G_t + (1 - \beta) (\nabla L(w_t))^2 $$
      $$ \Delta w_{t+1} = - \frac{\sqrt{\Delta w_t + \epsilon}}{\sqrt{G_{t+1}} + \epsilon} \nabla L(w_t) $$
      $$ w_{t+1} = w_t + \Delta w_{t+1} $$

- [x] adam (Adaptive Moment Estimation)
      $$ m_{t+1} = \beta_1 m_t + (1 - \beta_1) \nabla L(w_t) $$
      $$ v_{t+1} = \beta_2 v_t + (1 - \beta_2) (\nabla L(w_t))^2 $$
      $$ \hat{m}_{t+1} = \frac{m_{t+1}}{1 - \beta_1^{t+1}} $$
      $$ \hat{v}_{t+1} = \frac{v_{t+1}}{1 - \beta_2^{t+1}} $$
      $$ w_{t+1} = w_t - \frac{\eta}{\sqrt{\hat{v}_{t+1}} + \epsilon} \hat{m}\_{t+1} $$

- [x] adamax (Adam with Infinity Norm)
      $$ m_{t+1} = \beta_1 m_t + (1 - \beta_1) \nabla L(w_t) $$
      $$ u_{t+1} = \max(\beta_2 u_t, |\nabla L(w_t)|) $$
      $$ w_{t+1} = w_t - \frac{\eta}{u_{t+1}} m\_{t+1} $$

- [x] nadam (Nesterov Adam)
      $$ m_{t+1} = \beta_1 m_t + (1 - \beta_1) \nabla L(w_t) $$
      $$ v_{t+1} = \beta_2 v_t + (1 - \beta_2) (\nabla L(w_t))^2 $$
      $$ \hat{m}_{t+1} = \frac{\beta_1 m_{t+1} + (1 - \beta_1) \nabla L(w_t)}{1 - \beta_1^{t+1}} $$
      $$ \hat{v}_{t+1} = \frac{\beta_2 v_{t+1}}{1 - \beta_2^{t+1}} $$
      $$ w_{t+1} = w_t - \frac{\eta}{\sqrt{\hat{v}_{t+1}} + \epsilon} \hat{m}\_{t+1} $$

### Weight Initializers:

- [x] random (Random Normal)
- [x] xavier_normal (Xavier Normal Initialization)
- [x] xavier_uniform (Xavier Uniform Initialization)
- [x] he_normal (He Normal Initialization)
- [x] he_uniform (He Uniform Initialization)

## Report:

The complete report, including experiment results and analysis, has been generated using Wandb. You can access it here.

## GitHub Repository:

The code for this project is available on GitHub. You can find it here.

Note: Please adhere to academic integrity policies and refrain from collaborating or discussing the assignment with other students.
