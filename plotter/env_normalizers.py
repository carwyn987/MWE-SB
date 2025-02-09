def pendulum_v1_normalizer(returns: list):
        """
        Normalized returns
        NOTE: importantly, the decision was made to normalize with respect to the maximum possible reward, and the random behavior reward.
        """
        #min_reward_possible = 200 * -16.2736044 # timesteps * minimum reward possible
        random_behavior_reward = 200 * -10
        max_possible_reward = 0.0 # being perfectly balanced upwards, with no applied torque

        return [(x-random_behavior_reward)/(max_possible_reward-random_behavior_reward) for x in returns]