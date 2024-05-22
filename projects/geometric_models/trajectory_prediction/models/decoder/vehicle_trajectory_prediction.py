from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Generic, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch_geometric.nn.models import MLP

from commonroad_geometric.common.config import Config
from commonroad_geometric.learning.geometric.base_geometric import BaseModel
from projects.geometric_models.drivable_area.models.decoder.vehicle_model import T_VehicleStates, VehicleModel
from projects.geometric_models.drivable_area.models.modules.cvae import ConditionalVariationalAutoencoder
from projects.geometric_models.drivable_area.models.modules.time2vec import Time2Vec
from projects.geometric_models.drivable_area.models.modules.transformer_decoder import PositionalEncoding


class VehicleTrajectoryPredictionDecoder(BaseModel, Generic[T_VehicleStates]):

    def __init__(self, cfg: Config, vehicle_model: VehicleModel[T_VehicleStates], num_predict_time_steps: int):
        super().__init__()
        self.cfg = cfg
        self.vehicle_model = vehicle_model
        self.num_predict_time_steps = num_predict_time_steps

        if self.cfg.trajectory_prediction.decoder_type == "GRU":
            self.rnn_encoder = nn.GRU(
                input_size=cfg.traffic.vehicle_node_feature_size,
                hidden_size=self.cfg.trajectory_prediction.rnn.hidden_size,
                batch_first=True,
            )
            self.rnn_decoder_time2vec = Time2Vec(self.cfg.trajectory_prediction.rnn.time2vec_dim)
            self.rnn_decoder = nn.GRU(
                input_size=self.cfg.trajectory_prediction.rnn.time2vec_dim,
                hidden_size=self.cfg.trajectory_prediction.rnn.hidden_size,
                batch_first=False,
            )
            self.rnn_project_lin = nn.Linear(
                self.cfg.trajectory_prediction.rnn.hidden_size,
                vehicle_model.num_input_dims)

        elif self.cfg.trajectory_prediction.decoder_type == "Transformer":
            cfg_transformer = self.cfg.trajectory_prediction.transformer
            self.positional_encoding = PositionalEncoding(
                dim=self.cfg.traffic.vehicle_node_feature_size,
                max_length=self.cfg.traffic.temporal.steps_observe,
                dropout=cfg_transformer.dropout,
            )
            self.vehicle_x_proj = nn.Linear(self.cfg.traffic.vehicle_node_feature_size, cfg_transformer.model_size)
            self.transformer_decoder = nn.TransformerDecoder(
                decoder_layer=nn.TransformerDecoderLayer(
                    d_model=cfg_transformer.model_size,
                    dim_feedforward=cfg_transformer.feedforward_size,
                    nhead=cfg_transformer.attention_heads,
                    dropout=cfg_transformer.dropout,
                    batch_first=True,
                ),
                num_layers=cfg_transformer.num_layers,
            )
            # self.transformer_decoder = TransformerDecoderModel(
            #     num_layers=cfg_transformer.num_layers,
            #     d_model=cfg_transformer.model_size,
            #     d_feedforward=cfg_transformer.feedforward_size,
            #     heads=cfg_transformer.attention_heads,
            #     dropout=cfg_transformer.dropout,
            # )
            self.transformer_project_lin = nn.Linear(cfg_transformer.model_size, vehicle_model.num_input_dims)

        else:
            raise ValueError(f"Unknown decoder_type config value: {self.cfg.trajectory_prediction.decoder_type}")

    def forward(
        self,
        vehicle_x: Tensor,  # vehicle x time observed x vehicle embedding dim
        vehicle_states_last_obs: T_VehicleStates,
        dt: float,
        trajectory_targets: Optional[Tensor],
    ) -> List[T_VehicleStates]:
        N_veh, T_obs, D_vehicle = vehicle_x.size()
        T_pred = self.num_predict_time_steps
        device = self.device
        assert trajectory_targets is None
        trajectory_predictions = []

        if self.cfg.trajectory_prediction.decoder_type == "GRU":
            delta_times = dt * torch.arange(1, T_pred + 1, device=device)
            decoder_input = self.rnn_decoder_time2vec(delta_times.unsqueeze(-1))
            decoder_input = decoder_input.unsqueeze(1).repeat(1, N_veh, 1)
            # assert_size(decoder_input, (T_obs, N_veh, self.cfg.trajectory_prediction.rnn.time2vec_dim))

            _, hidden_state = self.rnn_encoder(vehicle_x)
            output, _ = self.rnn_decoder(decoder_input, hidden_state)
            predictions = self.rnn_project_lin(output)
            current_vehicle_states = vehicle_states_last_obs
            for t in range(T_pred):
                current_vehicle_states = self.vehicle_model.compute_next_state(
                    states=current_vehicle_states,
                    input=predictions[t],
                    dt=dt,
                )
                trajectory_predictions.append(current_vehicle_states)

        elif self.cfg.trajectory_prediction.decoder_type == "Transformer":
            memory_mask = torch.ones((1, T_pred + 1, T_obs), dtype=torch.bool, device=device)
            memory = self.vehicle_x_proj(self.positional_encoding(x=vehicle_x, mode="add"))
            preds = torch.zeros(
                (N_veh, 1, self.cfg.trajectory_prediction.transformer.model_size),
                dtype=torch.float32,
                device=device,
            )

            # TODO teacher forcing
            #   https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
            enable_teacher_forcing = random.random() < self.cfg.trajectory_prediction.teacher_forcing_rate  # TODO

            # TODO visualize attention patterns like here:
            # https://trajectory-transformer.github.io/ ("attention patterns")

            current_vehicle_states = vehicle_states_last_obs
            for t in range(T_pred):
                # out = self.transformer_decoder(
                #     predictions=preds,
                #     predictions_mask=subsequent_mask(preds.size(1)).to(device),
                #     memory=memory,
                #     memory_mask=memory_mask[:, :t + 1],
                # )
                # TODO check masking: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
                out = self.transformer_decoder(
                    tgt=preds,
                    tgt_mask=nn.Transformer.generate_square_subsequent_mask(preds.size(1)).to(device),
                    memory=memory,
                    memory_mask=memory_mask[:, :t + 1],
                )
                prediction = out[:, -1]
                preds = torch.cat([preds, prediction.unsqueeze(1)], dim=1)

                prediction = self.transformer_project_lin(prediction)
                current_vehicle_states = self.vehicle_model.compute_next_state(
                    states=current_vehicle_states,
                    input=prediction,
                    dt=dt,
                )
                trajectory_predictions.append(current_vehicle_states)

        return trajectory_predictions

    def compute_loss(
        self,
        prediction: Union[CVAETrainingOutput[T_VehicleStates], List[T_VehicleStates]],
        target_vehicle_states: List[T_VehicleStates],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        assert isinstance(prediction, list) and len(prediction) == len(target_vehicle_states)
        loss_sum = torch.tensor(0, dtype=torch.float32, device=self.device)
        for i, vehicle_states in enumerate(prediction):
            loss = self.vehicle_model.compute_single_step_loss(
                prediction=vehicle_states, target=target_vehicle_states[i])
            # TODO all other (non-primary) losses
            loss_sum = loss_sum + loss["primary"]
        return loss_sum / len(prediction), {}


@dataclass(frozen=True)
class CVAETrainingOutput(Generic[T_VehicleStates]):
    prediction: Tuple[Tensor, Tuple[Tensor, Tensor], Tensor, Tuple[Tensor, Tensor]]
    trajectory_targets: Tensor
    vehicle_states_last_obs: T_VehicleStates
    dt: float


class VehicleTrajectoryPredictionCVAEDecoder(BaseModel, Generic[T_VehicleStates]):

    def __init__(
        self,
        cfg: Config,
        vehicle_model: VehicleModel[T_VehicleStates],
        num_predict_time_steps: int,
        device: str
    ):
        super().__init__()
        self.cfg = cfg
        assert self.cfg.trajectory_prediction.decoder_type == "CVAE"
        self.vehicle_model = vehicle_model
        self.num_predict_time_steps = num_predict_time_steps

        recognition_network = CVAERecognitionNetwork(
            input_dims=self.cfg.traffic.vehicle_node_feature_size,
            target_dims=num_predict_time_steps * self.vehicle_model.num_state_dims,
            latent_dims=self.cfg.trajectory_prediction.cvae.latent_dims,
        )
        generator_network = CVAEGeneratorNetwork(
            input_dims=cfg.traffic.vehicle_node_feature_size,
            latent_dims=self.cfg.trajectory_prediction.cvae.latent_dims,
            output_dims_per_time_step=self.vehicle_model.num_input_dims,
            num_predict_time_steps=num_predict_time_steps,
        )
        # the generator network does not generate the final sample, we extract a deterministic transformation of the
        # generator output (computing the next state with the vehicle model) into the reconstruction loss
        # because doing so is more convenient
        self.cvae: ConditionalVariationalAutoencoder[Tensor, List[T_VehicleStates]] = ConditionalVariationalAutoencoder(
            recognition_network=recognition_network,
            prior_network=None,  # standard multivariate normal prior N(0, I)
            latent_dims=self.cfg.trajectory_prediction.cvae.latent_dims,
            generator_network=generator_network,
            device=device
        )

    def _compute_cvae_trajectory_prediction(
        self,
        initial_vehicle_states: T_VehicleStates,
        vehicle_model_input: Tensor,
        dt: float,
    ) -> List[T_VehicleStates]:
        trajectory_prediction = []
        current_vehicle_states = initial_vehicle_states
        for t in range(self.num_predict_time_steps):
            input = vehicle_model_input[:, t *
                                        self.vehicle_model.num_input_dims:(t + 1) * self.vehicle_model.num_input_dims]
            current_vehicle_states = self.vehicle_model.compute_next_state(
                states=current_vehicle_states,
                input=input,
                dt=dt,
            )
            trajectory_prediction.append(current_vehicle_states)
        return trajectory_prediction

    def forward(
        self,
        vehicle_x: Tensor,  # vehicle x time observed x vehicle embedding dim
        vehicle_states_last_obs: T_VehicleStates,
        dt: float,
        trajectory_targets: Optional[Tensor],
    ) -> Union[CVAETrainingOutput[T_VehicleStates], List[T_VehicleStates]]:
        N_veh, T_obs, D_vehicle = vehicle_x.size()
        assert self.training == (trajectory_targets is not None)

        # select vehicle representation from last observed time step as input to the CVAE
        vehicle_x = vehicle_x[:, T_obs - 1]
        # TODO don't use just the last time step?

        # TODO make it possible to get the expected inputs from the vehicle model
        # by inverting the compute_next_state method?

        prediction = self.cvae(
            input=vehicle_x,
            target=trajectory_targets,
        )
        if self.training:
            return CVAETrainingOutput(
                vehicle_states_last_obs=vehicle_states_last_obs,
                dt=dt,
                trajectory_targets=trajectory_targets,
                prediction=prediction,
            )

        else:
            trajectory_prediction = self._compute_cvae_trajectory_prediction(
                initial_vehicle_states=vehicle_states_last_obs,
                vehicle_model_input=prediction,
                dt=dt,
            )
            return trajectory_prediction

    def compute_loss(
        self,
        prediction: Union[CVAETrainingOutput[T_VehicleStates], List[T_VehicleStates]],
        target_vehicle_states: List[T_VehicleStates],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        if self.training:
            assert isinstance(prediction, CVAETrainingOutput)

            def reconstruction_loss(output: Tensor, target: List[T_VehicleStates]) -> Tensor:
                trajectory_prediction = self._compute_cvae_trajectory_prediction(
                    initial_vehicle_states=prediction.vehicle_states_last_obs,
                    vehicle_model_input=output,  # sample generated by CVAE
                    dt=prediction.dt,
                )
                assert len(trajectory_prediction) == len(target)
                loss_sum = torch.tensor(0, dtype=torch.float32, device=self.device)
                for i, vehicle_states in enumerate(trajectory_prediction):
                    loss = self.vehicle_model.compute_single_step_loss(prediction=vehicle_states, target=target[i])
                    # TODO all other (non-primary) losses
                    loss_sum = loss_sum + loss["primary"]
                return loss_sum / len(trajectory_prediction)

            return self.cvae.compute_hybrid_loss(
                prediction=prediction.prediction,
                target=target_vehicle_states,
                reconstruction_loss=reconstruction_loss,
                alpha=self.cfg.trajectory_prediction.cvae.hybrid_loss_alpha,
            )

        else:  # testing
            assert isinstance(prediction, list) and len(prediction) == len(target_vehicle_states)
            loss_sum = torch.tensor(0, dtype=torch.float32, device=self.device)
            for i, vehicle_states in enumerate(prediction):
                loss = self.vehicle_model.compute_single_step_loss(
                    prediction=vehicle_states, target=target_vehicle_states[i])
                # TODO all other (non-primary) losses
                loss_sum = loss_sum + loss["primary"]
            return loss_sum / len(prediction), {}


class CVAERecognitionNetwork(BaseModel):

    def __init__(self, input_dims: int, target_dims: int, latent_dims: int):
        super().__init__()
        self.latent_dims = latent_dims
        self.mlp = MLP(channel_list=[input_dims + target_dims, 256, 2 * latent_dims])

    def forward(
        self,
        input: Tensor,  # vehicle representation from last observed time step N_veh x D_veh
        target: Tensor,  # target trajectory N_veh x T_pred x D_vehicle_model_input
    ) -> Tuple[Tensor, Tensor]:
        N_veh = target.size(0)
        x = torch.cat([input, target.view(N_veh, -1)], dim=-1)
        params = self.mlp(x)
        mu, log_sigma = params[:, :self.latent_dims], params[:, self.latent_dims:]
        return mu, log_sigma


class CVAEGeneratorNetwork(nn.Module):

    def __init__(
        self,
        input_dims: int,
        latent_dims: int,
        output_dims_per_time_step: int,
        num_predict_time_steps: int,
    ):
        super().__init__()
        self.mlp = MLP(channel_list=[
            input_dims + latent_dims, 256, num_predict_time_steps * output_dims_per_time_step,
        ])

    def forward(self, input: Tensor, latent: Tensor) -> Tensor:
        if not self.training:
            # TODO: hack
            latent = latent.to(input.device)
        x = torch.cat([input, latent], dim=-1)
        vehicle_model_inputs = self.mlp(x)  # size: N_veh x (time steps * input size)
        return vehicle_model_inputs
