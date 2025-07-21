import gym

env = gym.make("Blackjack-v1")
env = gym.make("Taxi-v3")
env = gym.make("CartPole-v1")
env = gym.make("MountainCar-v0")
env = gym.make("FrozenLake-v1")

print(env.action_space)
print(env.observation_space)
