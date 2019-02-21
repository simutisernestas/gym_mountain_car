import gym
import random
import numpy as np
from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam


# Load environment
env = gym.make('MountainCar-v0')
# Game ends after 200 steps
goal_steps = 200


# Try random game, to see environment info and results
for step_index in range(200):
    # env.render() # render video game
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print("Step {}:".format(step_index))
    print("Action: {}".format(action))
    print("Observation: {}".format(observation))
    print("Reward: {}".format(reward))
    print("Done: {}".format(done))
    print("Info: {}".format(info))
    if done:
        break
            
env.reset()
env.close()


# Min score required to save training data
# Initial score is -1 per step, game have 200 steps
# So we need data which have higher score over game
min_score = -190
# Game count for collecting training data
intial_games = 10000

training_data = []
for game in range(intial_games):
    score = 0
    game_memory = []
    previous_observation = []
    for step in range(200):
        # env.render() # do not need for rendering as it would be very slow
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        
        if len(previous_observation) > 0: # skip this for the first iteration
            arr = np.zeros(3, dtype='int')      # one hot vector [0,0,0]
            np.put(arr, action, 1) # place the one with action e.g. [0,1,0]
            game_memory.append([previous_observation, arr]) # insert to memory [prev_position, action_taken]
                        
        previous_observation = observation
        if observation[0] > -0.2:  # if x coordinate is closer to the top of the hill
            reward = 1
        
        score += reward # 1 or -1
            
        if done:
            break
            
    if score >= min_score:
        # print(score)
        training_data = training_data + game_memory
            
    env.reset()   # start over again
    # env.close() # do not need for rendering as it would be very slow


print(f'Data points count:\n{len(training_data)}\nExample data point:\n{training_data[0]}')
print(f'Position:\n{training_data[0][0]}\nAction taken:\n{training_data[0][1]}')


# Build the model for learning what action to take in specific position
input_size = 2 # Position has 2 coordinates
output_size = 3 # Action is one hot encoded vector [0,1,0]

model = Sequential()
model.add(Dense(521, input_dim=input_size, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(output_size, activation='linear'))
model.compile(loss='mse', optimizer=Adam())


# Data preparation for model input
X = np.array([i[0] for i in training_data])
y = np.array([i[1] for i in training_data])


# Train the model
model.fit(X, y, epochs=3)


# Test model performance with visual render! :)
for each_game in range(10):
    score = 0
    game_memory = []
    prev_obs = []
    for step_index in range(goal_steps):
        env.render()
        
        if len(prev_obs) == 0:
            prev_obs = new_observation
            continue
        
        action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs)))[0])  
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        
        if done:
            break
            
    env.reset()
    scores.append(score)
    
env.close()

