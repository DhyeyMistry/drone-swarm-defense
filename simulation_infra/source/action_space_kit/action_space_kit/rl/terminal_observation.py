"""Capture final observations before DirectMARLEnv resets completed environments."""


class TerminalObservationMixin:
    """Place before DirectMARLEnv in the MRO; observations must be read-only.

    Isaac computes dones and rewards before calling _reset_idx, but returns observations
    after resetting. SB3 needs both states, especially for timeout value bootstrapping.
    """

    def step(self, actions):
        self.extras.pop("terminal_observations", None)
        self.extras.pop("episode_records", None)
        self.extras.pop("log", None)
        self._capture_terminal = True
        try:
            return super().step(actions)
        finally:
            self._capture_terminal = False

    def _reset_idx(self, env_ids):
        if getattr(self, "_capture_terminal", False):
            observe = getattr(self, "_get_terminal_observations", self._get_observations)
            self.extras["terminal_observations"] = {
                "env_ids": env_ids.clone(),
                "observations": {k: v[env_ids].clone() for k, v in observe().items()},
            }
        super()._reset_idx(env_ids)
