from mushroom_rl.algorithms.policy_search.policy_gradient import PolicyGradient


class StatefulPolicyGradient(PolicyGradient):

    def fit(self, dataset, **info):
        J = list()
        self.df = 1.
        self.J_episode = 0.
        self._init_update()
        for sample in dataset:
            x, u, r, xn, _, last = self._parse(sample)
            self._step_update(x, u, r, xn)
            self.J_episode += self.df * r
            self.df *= self.mdp_info.gamma

            if last:
                self._episode_end_update()
                J.append(self.J_episode)
                self.J_episode = 0.
                self.df = 1.
                self._init_update()

        assert len(J) > 1, "More than one episode is needed to compute the gradient"

        self._update_parameters(J)


class BatchedStatefulPolicyGradient(PolicyGradient):

    def fit(self, dataset, **info):
        self.df = 1.
        self.J_episode = 0.
        J = self._all_step_updates(dataset)

        assert len(J) > 1, "More than one episode is needed to compute the gradient"

        self._update_parameters(J)
