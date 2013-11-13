__author__ = "Ian Goodfellow"

class Algorithm(object):
    """
    Note: this seems like it might not be a good piece of the RL software
    ecosystem. For RL, the choice of algorithm seems more tied up with the
    choice of what kind of model you make, so it might make more sense to
    have the algorithm just be part of the Agent class. This implementation
    basically does nothing but query the agent for the learning algo.
    """

    def setup(self, agent, environment):
        self.action_func = environment.get_action_func()
        self.decide_func = agent.get_decide_func()
        self.learn_func = agent.get_learn_func()
        agent.reward_record = []
        self.agent = agent

    def train(self):
        a = self.decide_func()
        r = self.action_func(a)
        self.agent.reward_record.append(r)
        self.learn_func(a, r)

