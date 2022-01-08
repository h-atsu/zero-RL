import gym
import time

env = gym.make('CartPole-v0')
state = env.reset()
done = False


while not done:
    env.render()
    time.sleep(0.1)
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)

env.close()
