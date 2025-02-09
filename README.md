# SDRL Environment and Training Pipeline
## 1. Structure
The files are structured as follows:
```bash
+-- SDRL
(Training)
|   +-- main_train_ppo.py [Training script using Proximal Policy Optimization (PPO)]
|   +-- main_train_grad.py [Training script using gradient-based optimization]
(Testing)
|   +-- main_test_render.py [Script to test and render the trained model]
|   +-- main_test_throught.py [Script to test throughput performance]
(Environment)
|   +-- utils_isaac_drive_env.py [Implementation of the Drive Environment]
(Agents)
|   +-- utils_agent.py [Implementation of different agent models]
(Installation)
|   +-- README.md [Installation instructions]
```

## 2. Installation
### 2.1 Dependencies
```bash
Python 3.8
PyTorch (with CUDA support, if using GPU)
NumPy
tqdm
WandB
Gym
```

### 2.2 Installation

1. Create a Conda Environment:

```bash
conda create -n SDRL python==3.8 -y
conda activate SDRL
```

2. Install PyTorch:

For GPU:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

For CPU:
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

Install Other Dependencies:

```bash
pip install matplotlib
pip install tqdm
pip install wandb
pip install gym
```

## 3. Training

(Due to the fact that the related research paper is currently under review, the complete training data has not been made public yet. More data will be released upon the publication of the paper.)

### 3.1 Training with PPO

To train the model using Proximal Policy Optimization (PPO), run the following command:
```bash
python ./Training/main_train_ppo.py
```

### 3.2 Training with Gradient-Based Optimization

To train the model using gradient-based optimization, run the following command:
```bash
python ./Training/main_train_grad.py
```

## 4. Testing
### 4.1 Test and Render
To test the trained model and render the results, run the following command:
```bash
python ./Testing/main_test_render.py
```

### 4.2 Test Throughput
To test the throughput performance of the trained model, run the following command:

```bash
python ./Testing/main_test_throught.py
```

## 5. Environment

The utils_isaac_drive_env.py file implements the Drive Environment, which simulates the driving scenario and provides the necessary interfaces for training and testing.

## 6. Agents
The utils_agent.py file contains the implementation of different agent models, including:
Agent: Basic agent model.
AgentAcceleration: Agent model with acceleration-based control.
AgentVehicleDynamic: Agent model with vehicle dynamics.

## 7. Usage
PPO Training:
Modify the main_train_ppo.py file to set the desired hyperparameters.
Run the training script:
```bash
python ./Training/main_train_ppo.py
```
Gradient-Based Training:
Modify the main_train_grad.py file to set the desired hyperparameters.
Run the training script:
```bash
python ./Training/main_train_grad.py
```
Render Test:
Run the render test script:
```bash
python ./Testing/main_test_render.py
```
This will visualize the agent's performance in the Drive Environment.
Throughput Test:
Run the throughput test script:
```bash
python ./Testing/main_test_throught.py
```
This will measure the performance of the agent in terms of throughput.

## 8. Collected Data Display
To visualize the collected data and results, you can use the following tools:

### Matplotlib: 
For basic plotting and visualization.

### WandB: 
For logging and visualizing training metrics.

## 9. Expected Output
### 9.1 Training
The training process will log metrics such as reward, loss, and KL divergence.

The trained model will be saved to the data/interim/ directory.

### 9.2 Testing
The test process will render the agent's performance in the Drive Environment.

The throughput test will output the average throughput performance.