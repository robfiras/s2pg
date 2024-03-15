import numpy as np
from mushroom_rl.utils.preprocessors import StandardizationPreprocessor


class StandardizationPreprocessor_ext(StandardizationPreprocessor):

    def call_without_updating_stats(self, obs):

        norm_obs = np.clip(
            (obs - self._obs_runstand.mean) / self._obs_runstand.std,
            -self._clip_obs, self._clip_obs
        )

        return norm_obs
