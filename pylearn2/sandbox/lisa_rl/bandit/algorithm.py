__author__ = "Ian Goodfellow"

class Algorithm(object):
    """
    Bare-bones algorithm for driving a bandit learning problem.
    """

    def setup(self, agent, environment):
        self.decide_func = agent.get_decide_func()
        self.action_func = environment.get_action_func()
        self.learn_func = agent.get_learn_func()
        agent.reward_record = []
        self.agent = agent

    def train(self):
        a = self.decide_func()
        r = self.action_func(a)
        self.agent.reward_record.append(r)
        self.learn_func(a, r)

class ContextualBanditAlgorithm(Algorithm):
    """
    Bare-bones algorithm for driving a contextual bandit learning problem.
    """

    def setup(self, agent, environment):
        self.context_func = environment.get_context_func()
        self.decide_func = agent.get_decide_func()
        self.action_func = environment.get_action_func()
        self.learn_func = agent.get_learn_func()
        agent.reward_record = []
        self.agent = agent
        self.agent.dataset_yaml_src = environment.dataset.yaml_src

    def train(self):
        # TODO: this could all be much more efficient on GPU if s, a, and r
        # were stored in shared variables that all the different functions
        # are aware of.
        s = self.context_func()
        a = self.decide_func(s)
        r = self.action_func(a)
        # TODO: figure out how to remove waste here, where forward prop is
        # done a second time in order to do backprop
        self.learn_func(s, a, r)
        if r.ndim > 0:
            r = r.mean()
        self.agent.reward_record.append(r)
