# Reinforcement-Learning-with-Flappy-Bird

## Game Engine:
In Engine folder, we  have the final openAI game engine called flappy_env.py

## External Submodule
- **gym-simpleflappy**: A Gym environment for a Flappy bird clone with a simple observation space.

- **Observations returns**:
```
bird height, bird speed, distance to next pipe, height of next pipe, distance to second pipe, height of second pipe]
```

	- Use `gym.make("SimpleFlappy-v0")` for rewards only when a pipe is passed.

	- Use `gym.make("SimpleFlappyDistance-v0")` for rewards based on steps survived - less sparse rewards should make the task a bit easier.

- Installation
```
cd gym-simpleflappy
pip install -e .
```

- Example
	- Run this sample agent to see the environment in action:
	```bash
	python -m gym_simpleflappy.examples.flappy_random_agent
	```
	- This agent picks a random action (flap or not flap) every 15 frames. Score is printed after each episode.