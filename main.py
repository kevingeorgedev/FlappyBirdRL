import flappy_bird_gymnasium
import gymnasium
import torch
from agent import Agent
import random

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

env = gymnasium.make("FlappyBird-v0", use_lidar=False) # , render_mode="human"
n_actions = env.action_space.n
n_observations = env.observation_space.shape[0]
print(f"Action space: {n_actions}")
print(f"Observation space: {n_observations}")

NUM_EPISODES = 400
GAMMA = 0.99
EXPLORE = 20000
INITIAL_EPSILON = 0.1
FINAL_EPSILON = 0.0001
REPLAY_MEMORY = 1,000,000
BATCH = 256
LR = 3e-4
TRAIN_AFTER = 1024
USE_ATTENTION = True
MOVING_AVERAGE = 100
SEQUENCE_LENGTH = 128

UPDATE_STEPS = 100
train_mode = True
learn_steps = 0

epsilon = INITIAL_EPSILON

#state, _ = env.reset()

agent = Agent(lr=LR, n_obs=n_observations, n_actions=n_actions, gamma=GAMMA, device=device, capacity=REPLAY_MEMORY, seq_length=SEQUENCE_LENGTH)

for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    done = False
    reward_e = 0

    while True:
        env.render()
        if random.random() < epsilon:
            action = env.action_space.sample() #random.choice([0, 1])
        else:
            tensor_state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
            action = agent.policy.get_action(tensor_state)

        next_state, reward, done, _, info = env.step(action)
        reward_e += reward

        agent.replay.add(state, action, next_state, reward, done)
        if len(agent.replay) > TRAIN_AFTER:
            if train_mode:
                print("starting training...")
                train_mode = False
            learn_steps += 1
            if learn_steps % UPDATE_STEPS == 0:
                agent.target.load_state_dict(agent.policy.state_dict())

            batch = agent.replay.sample(batch_size=BATCH, sequence_len=SEQUENCE_LENGTH)
            batch_state, batch_action, batch_next_state, batch_reward, batch_done = zip(*batch)

            batch_state = torch.tensor(batch_state, dtype=torch.float32, device=device)
            batch_next_state = torch.tensor(batch_next_state, dtype=torch.float32, device=device)
            batch_action = torch.tensor(batch_action, dtype=torch.float32, device=device).unsqueeze(1)
            batch_reward = torch.tensor(batch_reward, dtype=torch.float32, device=device).unsqueeze(1)
            batch_done = torch.tensor(batch_done, dtype=torch.float32, device=device).unsqueeze(1)

            with torch.no_grad():
                policyQ_next = agent.policy(batch_next_state)
                targetQ_next = agent.target(batch_next_state)
                policy_max_action = torch.argmax(policyQ_next, dim=1, keepdim=True)
                y = batch_reward + (1 - batch_done) * GAMMA * targetQ_next.gather(1, policy_max_action.long())

            loss = agent.criterion(agent.policy(batch_state).gather(1, batch_action.long()), y)
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()

            if epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
        
        if done:
            break

        state = next_state

    print(f"Episode: {episode + 1}, Score: {round(reward_e, 2)}")

while True:
    # Next action:
    # (feed the observation to your agent here)
    action = env.action_space.sample()

    # Processing:
    state, reward, done, _, info = env.step(action)
    
    # Checking if the player is still alive
    if done:
        break

env.close()