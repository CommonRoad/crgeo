from typing import Callable, Dict, List, Optional

from commonroad_geometric.learning.geometric.training.callbacks.base_callback import BaseCallback, StepCallbackParams


class DebugTrainBackwardGradientsCallback(BaseCallback[StepCallbackParams]):
    def __init__(
        self,
        frequency: int,
        return_gradients: bool = False,
        gradient_callback: Optional[Callable[[Dict, List[str]], None]] = None
    ):
        self.return_gradients = return_gradients
        self.gradient_callback = gradient_callback
        self.frequency = frequency
        self.call_count = 0

    def __call__(self, params: StepCallbackParams) -> None:
        # Calculating gradient debug info
        self.call_count += 1

        if self.call_count % self.frequency != 0:
            return

        total_norm = 0
        info = dict(gradients={})
        warnings = []
        for n, p in params.ctx.model.named_parameters():
            if p.grad is None:
                continue
            param_norm_grad = p.grad.data.norm(2).item()
            if not p.grad.isfinite().all():
                warnings.append(f"Param {n} with non-finite gradients and norm {param_norm_grad:.3e}")

            print(f"{n:<100}{param_norm_grad:.4e}")

            total_norm += param_norm_grad ** 2

            if self.return_gradients:
                p_grad = p.grad.detach()
                param_std_grad = p_grad.std().item()
                param_absmax_grad = p_grad.abs().max().item()

                p_tensor = p.T.detach()
                param_norm = p_tensor.data.norm(2).item()
                param_std = p_tensor.std().item()
                param_absmax = p_tensor.abs().max().item()
                info["gradients"][n] = (
                    p_grad.shape.numel(), str(list(p_grad.shape)),
                    param_norm_grad, param_std_grad, param_absmax_grad,
                    param_norm, param_std, param_absmax,
                )
        total_norm = total_norm ** 0.5
        info['|g|'] = total_norm
        if self.gradient_callback is not None:
            self.gradient_callback(info, warnings)
