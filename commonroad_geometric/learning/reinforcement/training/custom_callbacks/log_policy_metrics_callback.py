import torch
from stable_baselines3.common.callbacks import BaseCallback

EPS = 1e-5


class LogPolicyMetricsCallback(BaseCallback):
    """
    Records debug info for policy network at the start of each new policy rollout.
    """

    def __init__(self, verbose: int = 0):
        super(LogPolicyMetricsCallback, self).__init__(verbose=verbose)
        self.num_rollouts = 0

    def _on_training_start(self) -> None:
        return

    def _on_training_end(self) -> None:
        return

    def _on_rollout_start(self) -> None:
        if self.num_timesteps == 0:
            return
        self.num_rollouts += 1
        for name, param in self.model.policy.named_parameters():
            grad = param._grad
            weights = param.data
            if grad is None:
                continue
            absweights = torch.abs(weights)
            absgrad = torch.abs(grad)
            
            #self.logger.record(f"gradients/{name}_mean", torch.mean(grad).item())
            self.logger.record(f"gradients/{name}_max", torch.max(grad).item())
            self.logger.record(f"gradients/{name}_min", torch.min(grad).item())
            self.logger.record(f"gradients/{name}_std", torch.std(grad).item())
            self.logger.record(f"gradients/{name}_absmean", torch.mean(absgrad).item())
            self.logger.record(f"gradients/{name}_absmax", torch.max(absgrad).item())
            #self.logger.record(f"gradients/{name}_absmin", torch.min(absgrad).item())
            self.logger.record(f"gradients/{name}_vanished", torch.mean((absgrad < EPS).float()).item())

            #self.logger.record(f"weights/{name}_mean", torch.mean(weights).item())
            self.logger.record(f"weights/{name}_max", torch.max(weights).item())
            self.logger.record(f"weights/{name}_min", torch.min(weights).item())
            self.logger.record(f"weights/{name}_std", torch.std(weights).item())
            self.logger.record(f"weights/{name}_absmean", torch.mean(absweights).item())
            self.logger.record(f"weights/{name}_absmax", torch.max(absweights).item())
            #self.logger.record(f"weights/{name}_absmin", torch.min(absweights).item())
            self.logger.record(f"weights/{name}_dead", torch.mean((absweights < EPS).float()).item())

            self.logger.record("info/n_rollouts", self.num_timesteps)

        self.logger.dump(step=self.num_timesteps)

    def _on_step(self) -> bool:
        return True
