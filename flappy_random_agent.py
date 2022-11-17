#example to show rendering of env using openAI gym game engine for random policy

import gym
import gym_simpleflappy

env = gym.make("SimpleFlappy-v0")
env.render(mode = 'human')
s = env.reset()



done = False
R = []

while not done:
	a = env.action_space.sample()
	s,r,done,info = env.step(a)
	
	R.append(r)
	env.render(mode='human')
print(s)
env.close()
#%%
env.reset()

max_eps = 1000
episode = 0

steps = 0

while episode < max_eps:
    obs = env.reset()
    steps = 0
    score = 0
    while True:

        if steps % 15 == 0:
            action = env.action_space.sample()
        else:
            action = 0

        obs, reward, done, _ = env.step(action)
        env.render(mode = 'human')
        score += reward
        steps += 1
        if done:
            break

    print(score)
    episode += 1

env.close()
