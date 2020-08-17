# Report

## 1. Deep Reinforcement Learning Architecture
In this project, the DDPG algorithm is implemented to solve the collaboration and competition task.
- Initialise a replay buffer (from the ReplayBuffer.py file) as the agents memory
- Replay buffer holds BUFFER_SIZE number of training exploration and samples BATCH_SIZE number of these for learning iteratively at each step.
- An online / local actor (policy) and critic (Q-value) network is initialized. 
- Soft updates are used for adjusting the weight matrices in the corresponding actor and critic target networks

### Training:
 - the agent is trained for n numbers of episodes with a max_t number of timesteps per one episode or ends before if reaching a terminal state (dones)
 - inside the training loop, the agent acts (randomly at the start), appends SARS' sequences to memory and learns from them
 
 ### Actor Architecture
 - input dimension = state_space = int 24
 - output dimension = action_space = int 2
 - network is composed of two hidden (with dimensions 200 and 150) and one final output layer (dimensions = actor space), all of them using nn.Linear()
 - after the input layer, batch normalization is applied
 - for hidden layer, activation is relu, for output layer a tanh activation follows
 
### Critic Architecture
 - input dimension = state_space = int 24; first hidden layer returns 200 outputs
 - in the first hidden layer (input layer), state and action spaces are concatenated -> it sees 24 + 2 (state_size + action_size) inputs
 - second hidden layer sees 200  number of inputs; outputs = 150 nodes
 - for hidden layers, relu activation followed, final layer returns non-activated output
 
## 2. Hyperparamters
```
# Network params
FC1_UNITS = 200         # first hidden layer number of nodes
FC2_UNITS = 150         # second hidden layer number of nodes

# Agent params
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic 
WEIGHT_DECAY = 0        # L2 weight decay

DELAY_UPDATE = 2        # target network update delay to ensure stabilisation before
NOISE_DECAY = 0.9999    # reduce noise

RANDOM_SEED = 0         # seed for reproducibility

# Noise params
MU = 0.                 # OUNoise parameter
THETA = 0.15            # OUNoise parameter
SIGMA = 0.2             # OUNoise parameter
SCALE = 1               # OUNoise parameter

# Training params       
NUM_EPISODES = 1000     # number of training iterations     
PRINT_EVERY = 100       # how often to print stats
```

## 3. Opinion
### Multi-Agent-Learning appears to be very unstable and several algorithms including PPO, and TD3 (that has shown suoerior results in many other tasks compared to DDPG) have failed while trying to implement them in this task. Setting the hyperparameters accordingly was the crucial point in this project, and auto-optimizing them with either grid search or services like Amazon Sagemaker's Hyperparameter Optimization Job might greatly enhance the performance of this model.

## 4. Outlook
### One mayor improvement might be to share observation spaces. Both agents critics might receive each opponents current observation space and (as this is a collaborative task) might help each other in finding an optimal action. As for instance agent1 is faced with an environment state that is has never seen before, agent2 might already have experienced this situation and share its knowlegde. Other improvements like expected would be to include TD3-Learning or other advancements over DDPG, use Eperience Replay Buffer etc (most of these improvements are implemented in the Rainbow model, see below) and to use Monte Carlo Tree Search as well.

Additional options
- [Rainbow](https://arxiv.org/pdf/1710.02298.pdf): Rainbow networks apply several improvements over the classical DQN network and might help improving performance of the Critic's Q-value estimation.
- [Monte Carlo Tree Search](https://science.sciencemag.org/content/362/6419/1140/tab-pdf): Currently, the TD3 model as implemented here only consides deciding on the next state based on the current state. In Monte Carlo Tree Search a parent tree with multiple leaf nodes can be created to expand looking into the future not only towards the next state, but towards n number of states in the future and deciding the current strategy based on possible favourable decisions that lay much further in the future.
