#Disclaimer: using RL is not the recommended way to achieve this, this is for edu purposes, the recommended is probably minimax or some info-theoretic search (the results are not well this is just to practice some RL in pytorch)
#But I don't mess with information theory that much.
#did minimax in joined file

import pandas as pd
import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque

#Part 0
#Building the Bank
path = "valid_guesses.csv"
df = pd.read_csv(path)
WordBank = []
WordBank = df["word"].tolist()
N = len(WordBank)

#Part 1
#Building the GYM
class WordleEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, word_bank):
        super(WordleEnv, self).__init__()
        self.word_bank = word_bank
        self.max_guesses = 6
        self.word_length = 5

        self.action_space = spaces.Discrete(len(word_bank)) #i let my word bank to be the total action space
        self.observation_space = spaces.Box(
            low=0,
            high=3,
            shape=(26, self.word_length),
            dtype=np.int8
        )
        self.reset() #start a new epsidoe
        #the obs space is what the agent look at, what are the results, the system work on 25*5 matrix for each guess.
        # 0-unknown 1-for real not here 2-for real here
        #for example if the word is apple and i guess agent then a = [2 0 0 0 0] (could be doubles)
        #b = [0 0 0 0 0] b wasnt a part of the guess she could still be anywhere
        #e = [0 0 1 0 0] e is yellow so its there but it cant be the middle one
        #n = [1 1 1 1 1] grey cant be anywhere

    def reset(self):
        self.target_word = np.random.choice(self.word_bank) #pick a random word for the game
        self.guesses = [] #guess history
        self.done = False #done = T if you guessed 6 times or guessed correctly
        self.obs_matrix = np.zeros((26, self.word_length), dtype=np.int8) #fill the array with 0, nothing is known
        return self.obs_matrix.copy()

    def step(self, action): #the step is the function for every guess
        word_guess = self.word_bank[action] #recieve an action which is just a number in a long lists of words
        self.guesses.append(word_guess) #add the guess
        feedback = self._compute_feedback(word_guess) #get the feedback
        self._update_obs_matrix(word_guess, feedback)#update the matrix according to the feedback

        reward = self._compute_reward(feedback) #cumpute reward (cum)
        self.done = (word_guess == self.target_word) or (len(self.guesses) >= self.max_guesses) #checking if the game is sone
        return self.obs_matrix.copy(), reward, self.done, {}

    #this conclude the results with 2 - correct placement 1 for yellow placement and 0 for grey placement
    def _compute_feedback(self, guess):
        feedback = [0] * self.word_length #create [0 0 0 0 0]
        target_chars = list(self.target_word) #making the guess and the target to lists for easier comparisons
        guess_chars = list(guess)

        for i in range(self.word_length): #check for correct letters
            if guess_chars[i] == target_chars[i]:
                feedback[i] = 2
                target_chars[i] = None
            #check for yellow letters
        for i in range(self.word_length):
            if feedback[i] == 0 and guess_chars[i] in target_chars:
                feedback[i] = 1
                target_chars[target_chars.index(guess_chars[i])] = None

        return feedback

    def _update_obs_matrix(self, guess, feedback):
        for i, letter in enumerate(guess):
            idx = ord(letter) - 97
            if feedback[i] == 2: #keeps the 2s as 2s
                self.obs_matrix[idx, i] = 2
            elif feedback[i] == 1: #turn the yellow into a 1 in this position
                if self.obs_matrix[idx, i] != 2:
                    self.obs_matrix[idx, i] = 1
            else: #turn the 0 into 1s across the letters array
                for pos in range(self.word_length):
                    if self.obs_matrix[idx, pos] != 2 and self.obs_matrix[idx, pos] != 1:
                        self.obs_matrix[idx, pos] = 3

    def _compute_reward(self, feedback):
        if all(f == 2 for f in feedback):
            print("Nigga")
            return 25
        reward = -1
        reward += sum([1 if f == 2 else 0.5 if f == 1 else 0 for f in feedback])
        return reward

    def render(self, mode='human'): #make it humanly readble
        print("word to guess:", self.target_word)
        for g in self.guesses:
            fb = self._compute_feedback(g)
            display = ""
            for ch, f in zip(g, fb):
                if f == 2:
                    display += f"[{ch}]"
                elif f == 1:
                    display += f"({ch})"
                else:
                    display += f" {ch} "
            print(display)
        print("guesses:", len(self.guesses))
        print("observation matrix:\n", self.obs_matrix)

#part 2
#Build the model
class Network(nn.Module): #fairly simple randomize the weights with 42 zera
  def __init__(self, state_size, action_size, seed=42):
    super(Network, self).__init__()
    self.seed = torch.manual_seed(seed)
    self.fc1 = nn.Linear(state_size, 512)
    self.fc2 = nn.Linear(512, 512)
    self.fc3 = nn.Linear(512, action_size)

  def forward(self, state):
    x = self.fc1(state)
    x = F.relu(x)
    x = self.fc2(x)
    x = F.relu(x) #DO NOT USE SOFTMAX GODAMN I AM SO STUPID THIS SHIT DIDNT WORK FOR 50 MINS AND IT BECAUSE OF THE SOFTMAX FUCKCKCKCKCKKCKCKC I AM FUCKING RETARDED
    #anyway softmax is for distrubtion i need raw Q values for rewards estimation
    return self.fc3(x)

#Part 3
#HPs and the exprience replay
#those are pretty classic values
learning_rate = 5e-4
minibatch_size = 100
discount_factor = 0.99
replay_buffer_size = int(1e5)
interpolation_parameter = 1e-3

#classic implementation of Replay
class ReplayMemory(object):

  def __init__(self, capacity):
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #using GPU for faster results
    self.capacity = capacity #max memory capacity
    self.memory = [] #current memory

  def push(self, event):#push event in memory works in FIFO
    self.memory.append(event)
    if len(self.memory) > self.capacity:
      del self.memory[0]

  def sample(self, batch_size): #dividing the exprience into its respective groups
    experiences = random.sample(self.memory, k = batch_size)
    states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self.device)
    actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(self.device)
    rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device)
    next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(self.device)
    dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
    return states, next_states, actions, rewards, dones

#Classic implementation of a DQN class

class Agent():
  def __init__(self, state_size, action_size):
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.state_size = state_size
    self.action_size = action_size
    self.local_qnetwork = Network(state_size, action_size).to(self.device)
    self.target_qnetwork = Network(state_size, action_size).to(self.device)
    self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr = learning_rate)
    self.memory = ReplayMemory(replay_buffer_size)
    self.t_step = 0

  def step(self, state, action, reward, next_state, done):
    self.memory.push((state, action, reward, next_state, done))
    self.t_step = (self.t_step + 1) % 4
    if self.t_step == 0:
      if len(self.memory.memory) > minibatch_size:
        experiences = self.memory.sample(100)
        self.learn(experiences, discount_factor)

  def act(self, state, epsilon = 0.):
    state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
    self.local_qnetwork.eval()
    with torch.no_grad():
      action_values = self.local_qnetwork(state)
    self.local_qnetwork.train()
    if random.random() > epsilon:
      return np.argmax(action_values.cpu().data.numpy())
    else:
      return random.choice(np.arange(self.action_size))

  def learn(self, experiences, discount_factor):
    states, next_states, actions, rewards, dones = experiences
    next_q_targets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
    q_targets = rewards + discount_factor * next_q_targets * (1 - dones)
    q_expected = self.local_qnetwork(states).gather(1, actions)
    loss = F.mse_loss(q_expected, q_targets)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    self.soft_update(self.local_qnetwork, self.target_qnetwork, interpolation_parameter)

  def soft_update(self, local_model, target_model, interpolation_parameter):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
      target_param.data.copy_(interpolation_parameter * local_param.data + (1.0 - interpolation_parameter) * target_param.data)

#agent initialization
state_size = 26*5
number_actions = N
agent = Agent(state_size, number_actions)

#Part 4
#Training, retards friendly messages
#again pretty standard training except 1 thing
number_episodes = 5000
maximum_number_timesteps_per_episode = 6
epsilon_starting_value  = 1.0
epsilon_ending_value  = 0.01
epsilon_decay_value  = 0.995
epsilon = epsilon_starting_value
scores_on_100_episodes = deque(maxlen = 100)
#instead of using abs score to beat will be stupid since some words are impossible to achieve high scores (needs 5 anagrams of the word)
#so the training stops when i achieve so solved ratio
env = WordleEnv(WordBank) #creating an instance for the class because of some weirdass errors
solved_count = 0

for episode in range(1, number_episodes + 1):
  steps_to_solve = []
  state = env.reset()
  state_flat= state.flatten()
  score = 0
  solved_this_episode = False
  for t in range(maximum_number_timesteps_per_episode):
    action = agent.act(state_flat, epsilon)
    next_state, reward, done, _ = env.step(action)
    next_state_flat = next_state.flatten()
    agent.step(state_flat, action, reward, next_state_flat, done)
    state_flat = next_state_flat
    score += reward

    if done:
        steps_to_solve.append(len(env.guesses))
        if len(env.guesses) < 6:
            solved_this_episode=True
        break
  if solved_this_episode:
      solved_count += 1
  scores_on_100_episodes.append(score)

  epsilon = max(epsilon_ending_value, epsilon_decay_value * epsilon)

  print('\repisode {}\taverage score: {:.2f}\tsolved rate: {:.2f}%\t guesses: {}'.format(
      episode, np.mean(scores_on_100_episodes), (solved_count / episode) * 100, np.mean(steps_to_solve)
  ), end="")
  if episode % 100 == 0:
      print('\repisode {}\taverage score: {:.2f}\tsolved rate: {:.2f}%\t guesses: {}'.format(
          episode, np.mean(scores_on_100_episodes), (solved_count / episode) * 100, np.mean(steps_to_solve)
      ))
  if episode >= 100:
      solved_rate = solved_count / episode
      if solved_rate >= 0.95 and np.mean(steps_to_solve) < 4:  # stop if agent guesses correctly 90% of the time
          print('\nenvironment solved in {:d} episodes!\tsolved Rate: {:.2f}%\t guesses: {}'.format(
              episode, solved_rate * 100, np.mean(steps_to_solve)
          ))
          torch.save(agent.local_qnetwork.state_dict(), 'StinkyBalls.pth')
          break
