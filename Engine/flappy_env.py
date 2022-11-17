#this is final env we will be using for flappy bird
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pygame
from pygame.locals import *

import numpy as np


class Pipe:

    def __init__(self, height, gap, pos):
        self.height = height
        self.gap = gap
        self.pos = pos
        self.distance_to_bird = 0
        self.scored = False

    def reset_pipe(self, height, x):
        self.pos = x
        self.height = height


class FlappyEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    screen_width = 288
    screen_height = 512

    bird_color = pygame.Color('#222222')
    background_color = pygame.Color('#abcdef')
    ground_color = pygame.Color('#993333')
    font_color = pygame.Color('#FFFFFF')

    fps_timer = pygame.time.Clock()
    max_FPS = 30


    # physics
    gravity = 1.0
    jump_force = -9.0
    max_speed = 10.0

    # bird
    bird_initial_x = screen_width * 0.4
    bird_initial_y = screen_height * 0.5
    bird_x = bird_initial_x
    bird_y = bird_initial_y
    bird_size = 24
    velocity = 0

    # world
    ground_y = screen_height * 0.9
    pipe_max = ground_y - screen_height * 0.1
    pipe_min = screen_height * 0.4
    pipe_gap = 100
    pipe_width = 52
    pipe_speed = -4

    next_gap = ( screen_width - pipe_width ) / 2
    pipe_initial_pos = screen_width
    pipe_next_initial_pos = screen_width + next_gap

    pipes = [Pipe(0, pipe_gap, pipe_initial_pos), Pipe(0, pipe_gap, pipe_next_initial_pos)]

    done = False

    # actions and observation space
    action_space = spaces.Discrete(2)

    ###
    # Observations - bird height, bird velocity,
    #  distance to next pipe, height of next pipe,
    #  distance to second pipe, height of second pipe
    ###
    observation_space = spaces.Box(
        np.array([0, -np.inf, -screen_width, 0, -screen_width, 0]),
        np.array([screen_height, np.inf, screen_width, screen_height, screen_width, 0]))

    def __init__(self):
        pygame.init()

        self.window = pygame.display.set_mode((self.screen_width, self.screen_height))
        self._seed()

    def _step(self, action):

        pygame.event.pump()
        reward = 0

        if not self.done:
            if action == 1:
                self.velocity += self.jump_force
            else:
                if self.velocity < self.max_speed:
                    self.velocity += self.gravity

            self.bird_y += self.velocity

            if self.bird_y >= self.ground_y:
                self.done = True

            for pipe in self.pipes:
                pipe.pos += self.pipe_speed
                if pipe.pos + self.pipe_width < self.bird_x:
                    if not pipe.scored:
                        pipe.scored = True
                        reward = 1.0
                else:
                    # Check collisions
                    if self.bird_x + self.bird_size > pipe.pos and self.bird_x < pipe.pos + self.pipe_width:
                        if self.bird_y + self.bird_size > pipe.height or self.bird_y < pipe.height - pipe.gap:
                            self.done = True

                if pipe.pos + self.pipe_gap <= 0:
                    pipe.reset_pipe(self.np_random.uniform(self.pipe_min, self.pipe_max), self.screen_width)
                pipe.distance_to_bird = self._bird_distance(pipe)

        # put together the game state
        state = self._build_state()

        return np.array(state), reward, self.done, {}

    def _build_state(self):
        pipes_sorted = sorted(self.pipes, key=lambda p: p.distance_to_bird)

        return (self.bird_y, self.velocity, pipes_sorted[0].distance_to_bird, pipes_sorted[0].height,
                 pipes_sorted[1].distance_to_bird, pipes_sorted[1].height)

    def _bird_distance(self, pipe):
        if pipe.pos + self.pipe_width < self.bird_x:
            return pipe.pos + self.screen_width + self.pipe_width - self.bird_x
        else:
            return pipe.pos - self.bird_x

    def _reset(self):
        self.done = False
        self.bird_x = self.bird_initial_x
        self.bird_y = self.bird_initial_y
        self.velocity = 0

        self.pipes[0].reset_pipe(self.np_random.uniform(self.pipe_min, self.pipe_max), self.screen_width)
        self.pipes[0].distance_to_bird = self._bird_distance(self.pipes[0])
        self.pipes[1].reset_pipe(self.np_random.uniform(self.pipe_min, self.pipe_max), self.screen_width + self.pipe_width + self.pipe_gap)
        self.pipes[1].distance_to_bird = self._bird_distance(self.pipes[1])

        return self._build_state()

    def _render(self, mode='human', close=False):

        self.window.fill(self.background_color)
        pygame.draw.rect(self.window, self.ground_color, (0, self.ground_y, self.screen_width, self.screen_height))
        for pipe in self.pipes:
            pygame.draw.rect(self.window, self.ground_color, (pipe.pos, pipe.height, self.pipe_width, self.screen_height - pipe.height))
            pygame.draw.rect(self.window, self.ground_color, (pipe.pos, 0, self.pipe_width, pipe.height - pipe.gap))

        pygame.draw.rect(self.window, self.bird_color, (self.bird_x, self.bird_y, self.bird_size, self.bird_size))
        pygame.display.update()
        self.fps_timer.tick(self.max_FPS)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class FlappyEnvDistance(FlappyEnv):

    def _step(self, action):
        reward = 0
        obs, _, done, info = super(FlappyEnvDistance, self)._step(action)
        if not done:
            reward += 1.0

        return obs, reward, done, info
