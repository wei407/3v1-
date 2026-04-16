import omni_drones.utils.scene as scene_utils
import torch

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec


class Capture3v1(IsaacEnv):
    def __init__(self, cfg, headless):
        self.num_pursuers = int(cfg.task.num_pursuers)
        self.num_evaders = int(cfg.task.num_evaders)
        self.capture_radius = float(cfg.task.capture_radius)
        self.arena_height = float(cfg.task.arena_height)
        self.init_spread = float(cfg.task.init_spread)
        self.safe_z_min = float(cfg.task.safe_z_min)
        self.boundary_xy = float(cfg.task.boundary_xy)

        if self.num_pursuers != 3 or self.num_evaders != 1:
            raise ValueError("Capture3v1 currently expects exactly 3 pursuers and 1 evader.")

        super().__init__(cfg, headless)

        self.drone.initialize()
        self.root_state_dim = 13

        self.stats["capture_rate"][:] = 0.0
        self.stats["min_distance"][:] = 0.0

    def _design_scene(self):
        drone_model_cfg = self.cfg.task.drone_model
        self.drone, self.controller = MultirotorBase.make(
            drone_model_cfg.name,
            drone_model_cfg.controller,
            device=self.device,
        )

        scene_utils.design_scene()

        base_layout = torch.tensor(
            [
                [0.0, -1.0, self.arena_height],
                [0.0, 1.0, self.arena_height],
                [-1.2, 0.0, self.arena_height],
                [2.5, 0.0, self.arena_height],
            ],
            dtype=torch.float32,
        )
        self.base_layout = base_layout.to(self.device)
        self.drone.spawn(translations=[tuple(v.tolist()) for v in base_layout])
        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        drone_state_dim = int(self.drone.state_spec.shape[0])
        obs_dim = drone_state_dim + 3 + 6
        central_dim = 4 * drone_state_dim

        self.observation_spec = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {
                        "observation": UnboundedContinuousTensorSpec(
                            (self.num_pursuers, obs_dim), device=self.device
                        ),
                        "observation_central": UnboundedContinuousTensorSpec(
                            (central_dim,), device=self.device
                        ),
                    }
                )
            }
        ).expand(self.num_envs).to(self.device)

        self.action_spec = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {
                        "action": torch.stack(
                            [self.drone.action_spec] * self.num_pursuers, dim=0
                        )
                    }
                )
            }
        ).expand(self.num_envs).to(self.device)

        self.reward_spec = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {
                        "reward": UnboundedContinuousTensorSpec(
                            (self.num_pursuers, 1), device=self.device
                        )
                    }
                )
            }
        ).expand(self.num_envs).to(self.device)

        self.agent_spec["drone"] = AgentSpec(
            "drone",
            self.num_pursuers,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "observation_central"),
        )

        stats_spec = CompositeSpec(
            {
                "return": UnboundedContinuousTensorSpec(self.num_pursuers, device=self.device),
                "episode_len": UnboundedContinuousTensorSpec(1, device=self.device),
                "capture_rate": UnboundedContinuousTensorSpec(1, device=self.device),
                "min_distance": UnboundedContinuousTensorSpec(1, device=self.device),
            }
        ).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return

        self.drone._reset_idx(env_ids)

        n = env_ids.numel()
        pos = self.base_layout.unsqueeze(0).expand(n, -1, -1).clone()
        noise = torch.zeros_like(pos)
        noise[..., :2] = self.init_spread * torch.randn_like(noise[..., :2])
        pos = pos + noise
        pos[..., 2] = self.arena_height

        rot = torch.zeros(n, self.num_pursuers + self.num_evaders, 4, device=self.device)
        rot[..., 0] = 1.0
        vel = torch.zeros(n, self.num_pursuers + self.num_evaders, 6, device=self.device)

        self.drone.set_world_poses(
            pos + self.envs_positions[env_ids].unsqueeze(1),
            rot,
            env_ids,
        )
        self.drone.set_velocities(vel, env_ids)
        self.stats[env_ids] = 0.0

    def _pre_sim_step(self, tensordict: TensorDictBase):
        pursuer_actions = tensordict[("agents", "action")]
        full_actions = torch.zeros(
            self.num_envs,
            self.num_pursuers + self.num_evaders,
            self.drone.action_spec.shape[-1],
            device=self.device,
        )
        full_actions[:, : self.num_pursuers] = pursuer_actions

        state = self.drone.get_state()
        root_state = state[..., : self.root_state_dim]
        pursuer_pos = state[:, : self.num_pursuers, :3]
        evader_pos = state[:, self.num_pursuers :, :3]

        center = pursuer_pos.mean(dim=1, keepdim=True)
        away = evader_pos - center
        away_xy = away[..., :2]
        away_norm = away_xy.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        away_dir_xy = away_xy / away_norm

        target_pos = evader_pos.clone()
        target_pos[..., :2] = evader_pos[..., :2] + 0.75 * away_dir_xy
        target_pos[..., 2] = self.arena_height

        evader_cmd = self.controller.compute(
            root_state[:, self.num_pursuers],
            target_pos=target_pos.squeeze(1),
        )
        full_actions[:, self.num_pursuers] = evader_cmd

        self.effort = self.drone.apply_action(full_actions)

    def _compute_state_and_obs(self):
        self.drone_states = self.drone.get_state()
        pursuer_state = self.drone_states[:, : self.num_pursuers]
        evader_state = self.drone_states[:, self.num_pursuers :]

        pursuer_pos = pursuer_state[..., :3]
        evader_pos = evader_state[..., :3]

        rel_pe = evader_pos.expand(-1, self.num_pursuers, -1) - pursuer_pos

        teammate_rel = []
        for i in range(self.num_pursuers):
            others = [j for j in range(self.num_pursuers) if j != i]
            teammate_rel.append(
                torch.cat(
                    [pursuer_pos[:, others[0]] - pursuer_pos[:, i], pursuer_pos[:, others[1]] - pursuer_pos[:, i]],
                    dim=-1,
                )
            )
        rel_pp = torch.stack(teammate_rel, dim=1)

        obs = torch.cat([pursuer_state, rel_pe, rel_pp], dim=-1)
        central = self.drone_states.reshape(self.num_envs, -1)

        self._min_dist = (pursuer_pos - evader_pos).norm(dim=-1).min(dim=1).values.unsqueeze(-1)

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                    "observation_central": central,
                },
                "stats": self.stats.clone(),
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        reward = torch.zeros(self.num_envs, self.num_pursuers, 1, device=self.device)

        captured = self._min_dist < self.capture_radius
        too_low = (self.drone_states[..., 2] < self.safe_z_min).any(dim=1, keepdim=True)
        out_of_bounds = (self.drone_states[..., :2].abs() > self.boundary_xy).any(dim=-1).any(dim=1, keepdim=True)
        has_nan = torch.isnan(self.drone_states).any(dim=-1).any(dim=1, keepdim=True)

        terminated = captured | too_low | out_of_bounds | has_nan
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)

        self.stats["return"].add_(reward.squeeze(-1))
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(-1)
        self.stats["capture_rate"][:] = captured.float()
        self.stats["min_distance"][:] = self._min_dist

        return TensorDict(
            {
                "agents": {"reward": reward},
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )
