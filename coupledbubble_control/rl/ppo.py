from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn. functional as F
import torch.optim as optim
import time
import numpy as np
from pathlib import Path

from typing import Dict, Any, Union, Optional

from .common.memory import RolloutBuffer
from .common.writers import TFWriter
from .common.policies import ActorCriticGaussianPolicy, ActorCriticBetaPolicy
from .common.utils import process_final_observation


class PPO:
    metadata = {"hyperparameters" : ["pi_learning_rate", "vf_learning_rate", "ft_learning_rate", 
                                     "gamma", "gae_lambda", "mini_batch_size", "clip_coef", "clip_vloss", "ent_coef", "vf_coef", "max_grad_norm",
                                     "target_kl", "norm_adv", "num_envs", "rollout_steps", "num_update_epochs", "gradient_steps",
                                     "seed", "torch_deterministic", "cuda", "buffer_device", 
                                     "policy_type", "hidden_dims", "activations", "shared_dims"]}

    default_net_arch = {
        "hidden_dims": [126, 84],
        "activations": ["ReLU", "ReLU"],
        "shared_dims": 0}
    
    def __init__(
            self,
            venvs: Any,
            learning_rate: float = 2.5e-4,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            mini_batch_size: int = 256,
            clip_coef: float = 0.2,
            clip_vloss: bool = True,
            ent_coef: float = 0.01,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            target_kl: Optional[float] = None,
            norm_adv: bool = True,
            rollout_steps: int = 32,
            num_update_epochs: int = 4,
            seed: int = 1,
            torch_deterministic: bool = True,
            cuda: bool = True,
            buffer_device: str = "cuda",
            net_archs: Dict = default_net_arch,
            policy_type: str = "Beta",
            ) -> None:
        
        self.device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
        # Seeds ------
        self.generator = torch.Generator(device=self.device)
        self.seed = seed
        if self.seed is not None:
            self.generator.manual_seed(self.seed)
        #torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = torch_deterministic

        # Attribures ---
        self.venvs = venvs
        self.num_envs = venvs.num_envs

        # Hyperparameters ---
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.norm_adv = norm_adv
        self.rollout_steps = rollout_steps
        self.mini_batch_size = mini_batch_size
        self.batch_size = self.num_envs * self.rollout_steps       # = buffer_size
        self.num_update_epochs = num_update_epochs
        self.iterations_per_epoch = self.iterations_per_epoch = self.batch_size // self.mini_batch_size
        self.gradient_steps = self.num_update_epochs * self.iterations_per_epoch
        self.buffer_device = buffer_device

        # Neural Netwokrs ---
        self.policy_type = policy_type
        self.hidden_dims = net_archs["hidden_dims"]
        self.activations = net_archs["activations"]
        self.shared_dims = net_archs["shared_dims"]

        action_high = venvs.action_space.space_data.high
        action_low  = venvs.action_space.space_data.low
        action_shape = venvs.action_space.space_data.shape
        observation_shape = venvs.observation_space.space_data.shape

        if self.policy_type == "Gaussian":
            self.policy = ActorCriticGaussianPolicy(
                observation_shape[0],
                action_shape[0],
                action_high=action_high,
                action_low=action_low,
                **net_archs
            ).to(self.device)

        elif self.policy_type == "Beta":
            self.policy = ActorCriticBetaPolicy(
                observation_shape[0],
                action_shape[0],
                action_high=action_high,
                action_low=action_low,
                **net_archs
            ).to(self.device)

        else:
            raise ValueError(f"Err: policy_type {self.policy_type} is not a valid policy. Please Choose a valid policy `Beta` or `Gaussian`")

        # Optimizer ---
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=self.learning_rate)

        print("----Policy----")
        print(self.policy)

        # Rollout buffer ---
        self.memory = RolloutBuffer(
            self.num_envs,
            self.rollout_steps,
            observation_shape,
            action_shape,
            self.buffer_device,
            dtype=torch.float32
        )

    def save_model(self, save_fname: str, save_dir: Path):
        save_dir.mkdir(parents=True, exist_ok=True)
        save_fname = str(save_dir / save_fname) + ".th"
        torch.save(self.policy.state_dict(), save_fname)


    def load_model(self, load_fname: str, load_dir: Path):
        if not load_dir.exists():
            raise ValueError(f"{str(load_dir)} does not exist!")
        load_fname = str(load_dir / load_fname) + ".th"
        policy_state_dict = torch.load(load_fname, weights_only=True)
        self.policy.load_state_dict(policy_state_dict)


    def predict(self, 
               total_timesteps: Optional[int] = None,
               total_episodes: Optional[int] = None,
               metrics_dir: Optional[Path] = None,
               metrics_fname: Optional[str] = None):

        if (total_timesteps is None) == (total_episodes is None):
            raise ValueError("Exactly one of 'total_timesteps' or 'total_episodes' must be specified.")
        
        self.policy.eval()
        # --- SAMPLING ---
        obs, _ = self.venvs.reset()
        global_step = 0
        episodes = 0
        episode_returns, episode_lengths = [], []

        while True:
            with torch.no_grad():
                actions, _, _, _ = self.policy(obs)
            
            obs, _, _, _, infos = self.venvs.step(actions)

            if 'final_observation' in infos.keys():
                for idx in range(infos['dones'].numel()):
                    episodes += 1
                    print(f"global_step={global_step}, episode_return={infos['episode_return'][idx]:0.3f}, episode_length={infos['episode_length'][idx]:0.0f}")
                    # Collect episode statistics
                    episode_returns.append(infos['episode_return'][idx].item())
                    episode_lengths.append(infos["episode_length"][idx].item())

            global_step += self.num_envs
            # Check for simulation ends
            if total_episodes is not None and episodes >= total_episodes:
                print("Max episodes reached")
                break

            elif total_timesteps is not None and global_step >= total_timesteps:
                print("Max timesteps reached")
                break

        # --  SAVE STATISTICS --
        if len(episode_returns) > 0 and len(episode_lengths) > 0:
            episode_returns = np.array(episode_returns, dtype=np.float32)
            episode_lengths = np.array(episode_lengths, dtype=np.float32)
            avg_return = np.mean(episode_returns)
            std_return = np.std(episode_returns)
            avg_length = np.mean(episode_lengths)
            std_length = np.std(episode_lengths)

            print("-------------------------------------------")
            print(f"Number of Episodes: {len(episode_returns):.0f}")
            print(f"Episode Returns: {avg_return:.3f}+/-{std_return:.3f}")
            print(f"Episode Lengths: {avg_length:.3f}+/-{std_length:.3f}")
            print("-------------------------------------------")

            if metrics_fname is not None:
                if metrics_dir is None:
                    raise ValueError("'metrics_dir' must be provided when 'metrics_fname' is set.")
                metrics_dir.mkdir(exist_ok=True, parents=True)
                path = metrics_dir / f"{metrics_fname}.csv"

                np.savetxt(
                    fname=path,
                    X = np.column_stack((episode_returns, episode_lengths)),
                    delimiter=",",
                    header="episode_returns,episode_lengths",
                    comments="",            # ne tegyen '#' jelet a header elé
                    fmt="%.6f"              # formátum mindkét oszlopra
                )
            
        self.venvs.close()


    def learn(self,
            total_timesteps: int,
            log_dir: Path,
            project_name: str,
            trial_name: str,
            log_frequency: int = 10):
        
        writer = None
        if all(v is not None for v in (log_dir, project_name, trial_name)):
            writer = TFWriter(log_dir=str(log_dir), 
                              project_name=project_name, 
                              trial_name=trial_name, model=self)
        else:
            print("Warning: logging is disabled.")

        self.policy.train()
        global_step = 0     # timesteps
        num_updates = 0     # nn update
        episodes    = 0
        train_loop  = 0     # number of training loops
        should_stop = False
        start_time = time.time()

        ep_returns = []
        ep_lengths = []

        obs, _ = self.venvs.reset(seed=self.seed)
        #dones = torch.zeros(self.num_envs, device=self.device)
        while global_step < total_timesteps:
            train_loop += 1
            # ALGO LOGIC: sampling, rollouts
            sampling_start = time.time()
            for step in range(self.rollout_steps):
                with torch.no_grad():
                    actions, logprobs, _, values = self.policy(obs)
                
                # Execute physics simulations and log train data
                next_obs, rewards, terminateds, time_outs, infos = self.venvs.step(actions)

                # Handle terminal observations
                real_next_obs = process_final_observation(next_obs, infos)
                if 'final_observation' in infos.keys():                        
                    for idx in range(infos['dones'].numel()):
                        episodes += 1
                        episode_return = infos['episode_return'][idx].item()
                        episode_length = infos['episode_length'][idx].item()
                        ep_returns.append(episode_return)
                        ep_lengths.append(episode_length)
                        #print(f"global_step={global_step}, episode_return={infos['episode_return'][idx]:0.3f}, episode_length={infos['episode_length'][idx]:0.0f}")
                        if writer is not None:
                            writer.add_scalar("charts/episode_return", episode_return, episodes)
                            writer.add_scalar("charts/episode_length", episode_length, episodes)

                    # Log
                    mean_return = sum(ep_returns) / len(ep_returns)
                    mean_length = sum(ep_lengths) / len(ep_lengths)
                    print(
                        f"step={global_step} | episodes={episodes} | "
                        f"return={mean_return:.3f} | length={mean_length:.0f}"
                    )
                    ep_returns.clear()
                    ep_lengths.clear()
                
                # Handle time-out --> add discounted future value for the reward
                idx = torch.where(time_outs)[0]
                with torch.no_grad():
                    real_next_values = self.policy.value(real_next_obs).view(-1)
                    if idx.numel() > 0:
                        rewards[idx] += (self.gamma * real_next_values[idx])

                # State transition
                dones = torch.logical_or(terminateds, time_outs)        # Trajectory ends

                # Store transition to memory
                self.memory.store_transitions(step, obs, actions, logprobs, rewards, values.view(-1), dones)

                # Increas counters
                global_step += self.num_envs
                obs = next_obs.clone()
            sampling_end = time.time()
            
            # GAE-estimate ---
            self.memory.compute_gae_estimate_(real_next_values, dones, self.gamma, self.gae_lambda)
            
            # ALGO LOGIC: training ---
            # Get the sample from the replay memory
            b_obs,          \
            b_logprobs,     \
            b_actions,      \
            b_advantages,   \
            b_returns,      \
            b_values, _     = self.memory.sample()

            clipfracs = []
            training_start = time.time()
            for _ in range(self.num_update_epochs):
                b_idx = torch.randperm(self.batch_size, generator=self.generator, device=self.device)
                for idx0 in range(0, self.batch_size, self.mini_batch_size):
                    idxN = min(idx0 + self.mini_batch_size, self.batch_size)
                    mb_idx = b_idx[idx0:idxN]
                    num_updates += 1

                    _, newlogprob, entropy, newvalues = self.policy(b_obs[mb_idx], b_actions[mb_idx])
                    logratio = newlogprob - b_logprobs[mb_idx]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # approx_kl http://joschu.net/blog/kl-approx.html
                        #approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                    # Normalize the advantages
                    mb_advantages = b_advantages[mb_idx]

                    if self.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Calculate the losses -----------
                    # Policy loss
                    pi_loss = torch.max(-mb_advantages * ratio,
                                        -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)).mean()
                    
                    # Value loss
                    newvalues = newvalues.view(-1)

                    if self.clip_vloss:
                        vf_loss_unclipped = (newvalues - b_returns[mb_idx])**2
                        vf_clipped = b_values[mb_idx] \
                                    + torch.clamp(newvalues - b_values[mb_idx],
                                                    -self.clip_coef,
                                                    +self.clip_coef)
                        vf_loss_clipped = (vf_clipped - b_returns[mb_idx])**2
                        vf_loss_max = torch.max(vf_loss_unclipped, vf_loss_clipped)
                        vf_loss = 0.5 * vf_loss_max.mean()
                    else:
                        vf_loss = 0.5 * ((newvalues - b_returns[mb_idx])**2).mean()

                    # Entropy loss
                    ent_loss = entropy.mean()

                    # Total loss function
                    loss = pi_loss - self.ent_coef * ent_loss + self.vf_coef * vf_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    if self.target_kl is not None and approx_kl > self.target_kl:
                        should_stop = True
            training_end = time.time()
            
            # Calculate the explained variance
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true) + 1.0e-12
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            if writer is not None and (train_loop % log_frequency == 0):
                writer.add_scalar("charts/learning_rate_pi", self.optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("charts/learning_rate_vf", self.optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("losses/vf_loss", vf_loss.item(), global_step)
                writer.add_scalar("losses/pi_loss", pi_loss.item(), global_step)
                writer.add_scalar("losses/ent_loss", ent_loss.item(), global_step)
                writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
                writer.add_scalar("accuracy/vf_values_mean", newvalues.mean().item(), global_step)
                writer.add_scalar("accuracy/approx_kl", approx_kl.item(), global_step)
                writer.add_scalar("accuracy/explained_variance", explained_var, global_step)

                sampling_time = sampling_end - sampling_start
                training_time = training_end - training_start

                writer.add_scalar("perf/training_time", training_time, global_step)
                writer.add_scalar("perf/sampling_time", sampling_time, global_step)
                writer.add_scalar("perf/DPS", float((global_step) / (time.time() - start_time)), global_step)

            if should_stop:
                print("Training stops!")
                break


        if writer is not None:
            writer.close()

        self.venvs.close()