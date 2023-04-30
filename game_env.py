import gym
class Environment:
    def __init__(self, game):
        env = gym.make(game, render_mode='human')
        env.reset()
        #For debbuging we use render
        for i in range(100):
            env.render()
            sample = env.action_space.sample()
            env.step(sample) 
        env.close()