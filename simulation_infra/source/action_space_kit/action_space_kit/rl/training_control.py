"""Stop at a PPO callback boundary so simulator callbacks cannot swallow cancellation."""

from stable_baselines3.common.callbacks import BaseCallback


class GracefulStopCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.requested = False

    def request_stop(self, *_):
        # A Python exception raised inside an Isaac callback may be logged and
        # swallowed. Record intent here; return from learning at the next step.
        self.requested = True

    def _on_step(self):
        if self.requested:
            print("[stop] Stop requested; returning from PPO to save model and normalization", flush=True)
        return not self.requested
