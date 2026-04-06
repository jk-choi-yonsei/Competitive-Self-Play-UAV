import argparse
import math
import random
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
from geopy.distance import distance
from geopy.point import Point
from haversine import haversine
from torch.utils.tensorboard import SummaryWriter

# Project setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = PROJECT_ROOT / "runs" / "PPO_Self_Play"
LOG_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

from ppo_agent.models import PPOActor, PPOCritic  # noqa: E402
from ppo_agent.rollout_buffer import RolloutBuffer  # noqa: E402
from ppo_agent.utils import get_action_ppo, eval_action_ppo  # noqa: E402
from envs import JSBSimF16ChaseEnv  # noqa: E402
from envs.target_controllers import bearing_deg, wrap180  # noqa: E402


class FrameStacker:
    """Maintains a fixed-length stack of frames."""

    def __init__(self, stack_size: int = 5):
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)

    def reset(self, frame: np.ndarray) -> None:
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(frame.astype(np.float32))

    def append(self, frame: np.ndarray) -> None:
        if not self.frames:
            self.reset(frame)
        else:
            self.frames.append(frame.astype(np.float32))

    def stacked(self) -> np.ndarray:
        if not self.frames:
            raise RuntimeError("FrameStacker is empty; call reset first.")
        arr = np.array(self.frames, dtype=np.float32)
        return np.expand_dims(arr, axis=0)


def state_from_fdm(fdm) -> np.ndarray:
    """Extract normalized state features from a JSBSim FDM."""
    alt_m = fdm.get_property_value("position/h-sl-meters")
    vals = [
        fdm.get_property_value("velocities/u-fps") / 500.0,
        fdm.get_property_value("velocities/v-fps") / 100.0,
        fdm.get_property_value("velocities/w-fps") / 100.0,
        alt_m / 10000.0,
        fdm.get_property_value("velocities/p-rad_sec") / 3.0,
        fdm.get_property_value("velocities/q-rad_sec") / 3.0,
        fdm.get_property_value("velocities/r-rad_sec") / 3.0,
        fdm.get_property_value("attitude/phi-deg") / 180.0,
        fdm.get_property_value("attitude/theta-deg") / 90.0,
        fdm.get_property_value("attitude/psi-deg") / 180.0,
        fdm.get_property_value("attitude/pitch-rad") / 1.57,
    ]
    return np.array(vals, dtype=np.float32)


def positional_geo_raw(primary, secondary) -> Tuple[float, float, float, float, float, float, float, float, float]:
    """
    Relative geometry from primary -> secondary.
    Returns: aspect_angle, angle_off, distance_2d_km, distance_3d_km, diff_lat, diff_lon, diff_alt_m, to_tgt_hdg, to_tgt_pitch
    """
    lat = float(primary.get_property_value("position/lat-gc-deg"))
    lon = float(primary.get_property_value("position/long-gc-deg"))
    alt_m = float(primary.get_property_value("position/h-sl-meters"))

    tgt_lat = float(secondary.get_property_value("position/lat-gc-deg"))
    tgt_lon = float(secondary.get_property_value("position/long-gc-deg"))
    tgt_alt = float(secondary.get_property_value("position/h-sl-meters"))

    diff_long = lon - tgt_lon + 1e-10
    diff_lat = lat - tgt_lat + 1e-10
    diff_alt = alt_m - tgt_alt

    to_tgt = math.degrees(math.atan(diff_lat / diff_long))
    if diff_long > 0 and diff_lat > 0:
        to_tgt_hdg = 270 - to_tgt
    elif diff_long < 0 < diff_lat:
        to_tgt_hdg = 90 - to_tgt
    elif diff_long > 0 > diff_lat:
        to_tgt_hdg = 270 - to_tgt
    else:
        to_tgt_hdg = to_tgt

    range_km = (haversine((lat, lon), (tgt_lat, tgt_lon)) * 1000) + 1e-6
    to_tgt_pitch = math.atan(diff_alt / range_km)

    point = Point(lat, lon)
    point_target = Point(tgt_lat, tgt_lon)
    distance_2d = distance(point, point_target).kilometers
    distance_3d = float(np.sqrt(distance_2d**2 + (alt_m / 100 - tgt_alt / 100) ** 2))

    heading = float(primary.get_property_value("attitude/psi-deg"))
    heading_target = float(secondary.get_property_value("attitude/psi-deg"))
    if heading > heading_target:
        hca = heading - heading_target
        angle_off = 360 - hca if hca > 180 else hca
    else:
        hca = heading_target - heading
        angle_off = 360 - hca if hca > 180 else hca

    tgt_x = math.sin(math.radians(heading_target))
    tgt_y = math.cos(math.radians(heading_target))
    long_back = tgt_lon - tgt_x
    lat_back = tgt_lat - tgt_y
    p1 = [long_back, lat_back]
    p2 = [tgt_lon, tgt_lat]
    p3 = [lon, lat]
    pt1 = (p1[0] - p2[0], p1[1] - p2[1])
    pt2 = (p3[0] - p2[0], p3[1] - p2[1])
    ang1 = math.atan2(pt1[1], pt1[0])
    ang2 = math.atan2(pt2[1], pt2[0])
    aspect = math.degrees(abs(ang1 - ang2))
    aspect_angle = aspect - (aspect - 180) * 2 if aspect > 180 else aspect
    ang1_deg = math.degrees(ang1)
    ang2_deg = math.degrees(ang2)
    if ang1_deg > 0:
        ang1_deg = -360 + ang1_deg
    if ang2_deg > 0:
        ang2_deg = -360 + ang2_deg
    if 0 >= ang1_deg >= -180:
        if ang1_deg > ang2_deg > ang1_deg - 180:
            aspect_angle *= -1
    else:
        if ang2_deg < ang1_deg or 0 > ang2_deg > ang1_deg + 180:
            aspect_angle *= -1

    return (
        aspect_angle,
        angle_off,
        distance_2d,
        distance_3d,
        diff_lat,
        diff_long,
        diff_alt,
        to_tgt_hdg,
        to_tgt_pitch,
    )


def positional_geo(primary, secondary) -> np.ndarray:
    raw = positional_geo_raw(primary, secondary)
    return np.array(
        [
            raw[0] / 180.0,
            raw[1] / 180.0,
            raw[2] / 50.0,
            raw[3] / 50.0,
            raw[4] / 0.1,
            raw[5] / 0.1,
            raw[6] / 1000.0,
            raw[7] / 180.0,
            raw[8] / 1.57,
        ],
        dtype=np.float32,
    )


def build_frame(env: JSBSimF16ChaseEnv, perspective: str = "agent") -> np.ndarray:
    """
    Build a single frame (20-dim) observation for the given perspective.
    perspective: "agent" -> agent relative to target, "target" -> target relative to agent.
    """
    if perspective == "target":
        primary, secondary = env.fdm_target, env.fdm
    else:
        primary, secondary = env.fdm, env.fdm_target
    state_vec = state_from_fdm(primary)
    geo_vec = positional_geo(primary, secondary)
    return np.hstack([state_vec, geo_vec]).astype(np.float32)


class EloManager:
    """Simple Elo rating manager with an anchor opponent."""

    def __init__(self, anchor_rating: float = 1000.0, k_factor: float = 32.0):
        self.anchor_rating = float(anchor_rating)
        self.agent_rating = float(anchor_rating)
        self.k_factor = float(k_factor)
        self.opponent_ratings: Dict[str, float] = {}

    def expected(self, ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def update(self, opponent_name: str, score_agent: float, opponent_is_anchor: bool = False) -> Tuple[float, float]:
        opp_rating = self.anchor_rating if opponent_is_anchor else self.opponent_ratings.get(opponent_name, self.anchor_rating)

        expected_agent = self.expected(self.agent_rating, opp_rating)
        expected_opp = 1.0 - expected_agent
        opp_score = 1.0 - score_agent

        self.agent_rating += self.k_factor * (score_agent - expected_agent)
        if not opponent_is_anchor:
            self.opponent_ratings[opponent_name] = opp_rating + self.k_factor * (opp_score - expected_opp)

        return self.agent_rating, self.anchor_rating if opponent_is_anchor else self.opponent_ratings[opponent_name]

    def set_opponent_rating(self, opponent_name: str, rating: float) -> None:
        self.opponent_ratings[opponent_name] = float(rating)


class BaseOpponent:
    def __init__(self, name: str, target_policy: str):
        self.name = name
        self.target_policy = target_policy

    @property
    def requires_action(self) -> bool:
        return False

    def reset(self, env: JSBSimF16ChaseEnv) -> None:
        return None

    def act(self, env: JSBSimF16ChaseEnv) -> np.ndarray:
        raise NotImplementedError


class PdOpponent(BaseOpponent):
    def __init__(self, name: str = "pd_anchor"):
        super().__init__(name=name, target_policy="pd")


class PastSelfOpponent(BaseOpponent):
    """Opponent that loads a past PPO actor checkpoint and drives the target (fixed mode)."""

    def __init__(
        self,
        name: str,
        checkpoint_path: Path,
        state_size: int,
        device: torch.device,
    ):
        super().__init__(name=name, target_policy="fixed")
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.actor = PPOActor(state_size, 4).to(device)
        state_dict = torch.load(checkpoint_path, map_location=device)["actor"]
        self.actor.load_state_dict(state_dict)
        self.actor.eval()
        self.stacker = FrameStacker(stack_size=5)

    @property
    def requires_action(self) -> bool:
        return True

    def reset(self, env: JSBSimF16ChaseEnv) -> None:
        frame = build_frame(env, perspective="target")
        self.stacker.reset(frame)

    def act(self, env: JSBSimF16ChaseEnv) -> np.ndarray:
        frame = build_frame(env, perspective="target")
        self.stacker.append(frame)
        obs = self.stacker.stacked()
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            mu, std = self.actor(obs_tensor)
            action_np, _ = get_action_ppo(mu, std)
        return action_np.flatten().astype(np.float32)


def ppo_update(
    rollout_buffer: RolloutBuffer,
    actor: PPOActor,
    critic: PPOCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    last_value: float,
    gamma: float,
    lam: float,
    clip_eps: float,
    ppo_epochs: int,
    mini_batch_size: int,
    entropy_coef: float,
    value_coef: float,
    max_grad_norm: float,
) -> Tuple[float, float, float]:
    """
    PPO clipped surrogate update.
    Computes GAE returns/advantages, then runs ppo_epochs of minibatch updates.
    Returns: (mean_actor_loss, mean_critic_loss, mean_entropy)
    """
    returns, advantages = rollout_buffer.compute_returns_and_advantages(last_value, gamma, lam)
    states_t, actions_t, old_log_probs_t, returns_t, advantages_t = rollout_buffer.get_tensors(returns, advantages)

    # Normalize advantages
    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    n = len(rollout_buffer)
    total_actor_loss = 0.0
    total_critic_loss = 0.0
    total_entropy = 0.0
    update_count = 0

    for _ in range(ppo_epochs):
        indices = torch.randperm(n)
        for start in range(0, n, mini_batch_size):
            end = min(start + mini_batch_size, n)
            mb_idx = indices[start:end]

            mb_states = states_t[mb_idx]
            mb_actions = actions_t[mb_idx]
            mb_old_log_probs = old_log_probs_t[mb_idx]
            mb_returns = returns_t[mb_idx]
            mb_advantages = advantages_t[mb_idx]

            # Actor forward
            mu, std = actor(mb_states)
            new_log_probs, entropy = eval_action_ppo(mu, std, mb_actions)

            # Ratio and clipped surrogate loss
            ratio = torch.exp(new_log_probs - mb_old_log_probs)
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_advantages
            actor_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy.mean()

            # Critic (value) loss
            values = critic(mb_states).squeeze(1)
            critic_loss = torch.nn.functional.mse_loss(values, mb_returns)

            # Combined update
            loss = actor_loss + value_coef * critic_loss

            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            actor_optimizer.step()
            critic_optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.mean().item()
            update_count += 1

    mean_actor = total_actor_loss / max(update_count, 1)
    mean_critic = total_critic_loss / max(update_count, 1)
    mean_entropy = total_entropy / max(update_count, 1)
    return mean_actor, mean_critic, mean_entropy


def get_winrate(stats: Dict[str, Dict[str, object]], name: str) -> float:
    s = stats.get(name)
    if s is None or s["games"] == 0:
        return 0.5
    return float(s["wins"]) / float(s["games"])


def select_opponent_mixed(
    pd_opponent: PdOpponent,
    past_opponents: Dict[str, PastSelfOpponent],
    self_play_ratio: float,
) -> BaseOpponent:
    """
    Select opponent based on self_play_ratio.
    - self_play_ratio prob: choose random PastSelfOpponent (if avail).
    - (1 - self_play_ratio) prob: choose PdOpponent.
    """
    if not past_opponents:
        return pd_opponent

    if random.random() < self_play_ratio:
        return random.choice(list(past_opponents.values()))
    else:
        return pd_opponent


def train(args):
    # Target PD configuration
    target_pd_config = {
        "k_heading_to_bank": args.pd_k_heading_to_bank,
        "k_elev_to_pitch": args.pd_k_elev_to_pitch,
        "k_bank_p": args.pd_k_bank_p,
        "k_bank_d": args.pd_k_bank_d,
        "k_pitch_p": args.pd_k_pitch_p,
        "k_pitch_d": args.pd_k_pitch_d,
        "k_rudder_damp": args.pd_k_rudder_damp,
        "max_bank_deg": args.pd_max_bank_deg,
        "max_pitch_cmd_deg": args.pd_max_pitch_cmd_deg,
        "aileron_sign": args.pd_aileron_sign,
        "elevator_sign": args.pd_elevator_sign,
        "throttle_cmd": args.pd_throttle,
    }

    env = JSBSimF16ChaseEnv(
        agent_steps=args.agent_steps,
        settle_steps=args.settle_steps,
        target_action=args.target_action,
        target_offset_lat=args.target_offset_lat,
        target_offset_long=args.target_offset_long,
        target_offset_alt=args.target_offset_alt,
        target_policy="pd",
        target_pd_config=target_pd_config,
        init_noise_config={
            "enabled": args.init_noise,
            "seed": args.init_noise_seed,
            "target_sigma_north_m": args.noise_target_sigma_north_m,
            "target_sigma_east_m": args.noise_target_sigma_east_m,
            "target_sigma_alt_m": args.noise_target_sigma_alt_m,
            "agent_sigma_u_fps": args.noise_agent_sigma_u_fps,
            "agent_heading_uniform_deg": args.noise_agent_heading_uniform_deg,
            "target_sigma_u_fps": args.noise_target_sigma_u_fps,
            "target_heading_uniform_deg": args.noise_target_heading_uniform_deg,
        },
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = args.model_name or run_name
    k = 0
    while True:
        tb_dir = LOG_DIR / f"{run_name}_{k:02d}"
        if not tb_dir.exists():
            break
        k += 1
    tb_dir.mkdir(parents=True, exist_ok=True)
    print(f"TensorBoard Logging to: {tb_dir}")

    writer = SummaryWriter(log_dir=str(tb_dir))
    fig_dir = FIG_DIR / model_name
    fig_dir.mkdir(parents=True, exist_ok=True)
    weight_dir = MODELS_DIR / model_name
    weight_dir.mkdir(parents=True, exist_ok=True)

    dummy_state = env.reset()
    state_shape = dummy_state.shape[1:]  # (time_steps, state_size)
    state_size = state_shape[1]
    action_size = 4

    actor = PPOActor(state_size, action_size).to(device)
    critic = PPOCritic(state_size).to(device)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    # Rollout buffer collects n_steps across episodes before each PPO update
    rollout_buffer = RolloutBuffer(args.n_steps * 2, state_shape, action_size, device)
    rollout_buffer.reset()

    # LR schedulers (optional linear decay with floor)
    def get_lr(base_lr: float, episode: int) -> float:
        if not args.lr_decay:
            return base_lr
        progress = min(episode / max(args.episodes - 1, 1), 1.0)
        return max(base_lr * (1.0 - progress) + args.min_lr * progress, args.min_lr)

    elo_mgr = EloManager(anchor_rating=1000.0, k_factor=args.elo_k_factor)
    pd_opponent = PdOpponent()
    past_opponents: Dict[str, PastSelfOpponent] = {}
    elo_history = []

    # Success-Triggered Gate State
    self_play_ratio = 0.0  # Starts at 0, only PD
    # trigger window fixed at 20 as per request
    trigger_window = 20
    recent_history = deque(maxlen=trigger_window)

    opponent_stats: Dict[str, Dict[str, object]] = {
        pd_opponent.name: {"wins": 0, "games": 0}
    }

    for episode in range(args.episodes):
        state = env.reset()
        done = False
        score = 0.0
        steps = 0
        mission_success = False
        last_action = None
        traj_lat, traj_lon, traj_alt = [], [], []
        traj_lat_tgt, traj_lon_tgt, traj_alt_tgt = [], [], []
        step_records = []

        # Entropy coef: linear decay over episodes
        current_entropy_coef = (
            args.entropy_coef
            - (args.entropy_coef - args.entropy_coef_end) * (episode / max(args.episodes - 1, 1))
        )

        # LR update (if decay enabled)
        if args.lr_decay:
            new_lr = get_lr(args.actor_lr, episode)
            for pg in actor_opt.param_groups:
                pg["lr"] = new_lr
            new_lr_c = get_lr(args.critic_lr, episode)
            for pg in critic_opt.param_groups:
                pg["lr"] = new_lr_c

        # 1. Opponent Selection
        opponent = select_opponent_mixed(pd_opponent, past_opponents, self_play_ratio)

        if opponent.target_policy == "pd":
            env.target_policy = "pd"
        else:
            env.target_policy = "fixed"
        opponent.reset(env)

        while not done and steps < args.max_steps:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
                mu, std = actor(state_tensor)
                action_np, log_prob = get_action_ppo(mu, std)
                value = critic(state_tensor).squeeze(-1).item()
                last_action = action_np

            if opponent.requires_action:
                tgt_action = opponent.act(env)
                env.target_action = tgt_action

            next_state, reward, done, info = env.step(action_np)
            if info.get("success", False):
                mission_success = True

            if args.save_step_data and (steps % args.step_data_stride == 0):
                t = float(env.fdm.get_property_value("simulation/sim-time-sec"))
                tgt_act = info.get("target_action", np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float32))
                step_records.append(
                    [
                        t,
                        float(reward),
                        1.0 if bool(info.get("success", False)) else 0.0,
                        float(info.get("range_3d_m", float("nan"))),
                        float(info.get("ata_deg", float("nan"))),
                        float(info.get("aot_deg", float("nan"))),
                        float(action_np.flatten()[0]),
                        float(action_np.flatten()[1]),
                        float(action_np.flatten()[2]),
                        float(action_np.flatten()[3]),
                        float(tgt_act[0]),
                        float(tgt_act[1]),
                        float(tgt_act[2]),
                        float(tgt_act[3]),
                        float(env.fdm.get_property_value("position/lat-gc-deg")),
                        float(env.fdm.get_property_value("position/long-gc-deg")),
                        float(env.fdm.get_property_value("position/h-sl-meters")),
                        float(env.fdm.get_property_value("attitude/psi-deg")),
                        float(env.fdm.get_property_value("attitude/theta-deg")),
                        float(env.fdm.get_property_value("attitude/phi-deg")),
                        float(env.fdm.get_property_value("velocities/u-fps")),
                        float(env.fdm.get_property_value("velocities/v-fps")),
                        float(env.fdm.get_property_value("velocities/w-fps")),
                        float(env.fdm.get_property_value("velocities/p-rad_sec")),
                        float(env.fdm.get_property_value("velocities/q-rad_sec")),
                        float(env.fdm.get_property_value("velocities/r-rad_sec")),
                        float(env.fdm_target.get_property_value("position/lat-gc-deg")),
                        float(env.fdm_target.get_property_value("position/long-gc-deg")),
                        float(env.fdm_target.get_property_value("position/h-sl-meters")),
                        float(env.fdm_target.get_property_value("attitude/psi-deg")),
                        float(env.fdm_target.get_property_value("attitude/theta-deg")),
                        float(env.fdm_target.get_property_value("attitude/phi-deg")),
                        float(env.fdm_target.get_property_value("velocities/u-fps")),
                        float(env.fdm_target.get_property_value("velocities/v-fps")),
                        float(env.fdm_target.get_property_value("velocities/w-fps")),
                        float(env.fdm_target.get_property_value("velocities/p-rad_sec")),
                        float(env.fdm_target.get_property_value("velocities/q-rad_sec")),
                        float(env.fdm_target.get_property_value("velocities/r-rad_sec")),
                    ]
                )

            lat, lon, alt_m = env.get_position()
            tgt_lat, tgt_lon, tgt_alt = env.get_target_position()
            traj_lat.append(lat)
            traj_lon.append(lon)
            traj_alt.append(alt_m)
            traj_lat_tgt.append(tgt_lat)
            traj_lon_tgt.append(tgt_lon)
            traj_alt_tgt.append(tgt_alt)

            # Push to rollout buffer (state without batch dim)
            rollout_buffer.push(state.squeeze(0), action_np.flatten(), log_prob, reward, done, value)
            state = next_state
            score += reward
            steps += 1

        # ----------------------------------------------------------------
        # PPO Update: trigger when rollout buffer has >= n_steps
        # ----------------------------------------------------------------
        last_actor_loss = None
        last_critic_loss = None
        last_entropy = None

        if len(rollout_buffer) >= args.n_steps:
            # Bootstrap value for last (non-terminal) state
            with torch.no_grad():
                last_state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
                last_val = critic(last_state_tensor).squeeze(-1).item() if not done else 0.0

            actor_loss, critic_loss, entropy = ppo_update(
                rollout_buffer=rollout_buffer,
                actor=actor,
                critic=critic,
                actor_optimizer=actor_opt,
                critic_optimizer=critic_opt,
                last_value=last_val,
                gamma=args.gamma,
                lam=args.lam,
                clip_eps=args.clip_eps,
                ppo_epochs=args.ppo_epochs,
                mini_batch_size=args.mini_batch_size,
                entropy_coef=current_entropy_coef,
                value_coef=args.value_coef,
                max_grad_norm=args.max_grad_norm,
            )
            last_actor_loss = actor_loss
            last_critic_loss = critic_loss
            last_entropy = entropy
            rollout_buffer.reset()  # Clear buffer after update

        # Plot trajectory
        try:
            import matplotlib.pyplot as plt  # Local import to avoid GUI issues

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(traj_lat, traj_lon, traj_alt, marker="o", markersize=2, alpha=0.8, label="Agent")
            ax.plot(
                traj_lat_tgt,
                traj_lon_tgt,
                traj_alt_tgt,
                marker="x",
                markersize=2,
                alpha=0.8,
                color="red",
                label="Target",
            )
            ax.scatter(
                traj_lat[0],
                traj_lon[0],
                traj_alt[0],
                s=200,
                c="blue",
                marker="o",
                edgecolors="black",
                linewidths=2,
                alpha=0.6,
                label="Agent Start",
            )
            ax.scatter(
                traj_lat_tgt[0],
                traj_lon_tgt[0],
                traj_alt_tgt[0],
                s=200,
                c="red",
                marker="o",
                edgecolors="black",
                linewidths=2,
                alpha=0.6,
                label="Target Start",
            )
            ax.set_xlabel("Latitude")
            ax.set_ylabel("Longitude")
            ax.set_zlabel("Altitude (m)")
            success_str = " [SUCCESS]" if mission_success else ""
            ax.set_title(f"Episode {episode} trajectory vs {opponent.name}{success_str}")
            ax.legend()
            fig.savefig(fig_dir / f"episode_{episode:05d}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"Failure: could not save figure for episode {episode}. Error: {e}")

        final_lat, final_lon, final_alt = env.get_position()
        tgt_lat, tgt_lon, tgt_alt = env.get_target_position()
        dist_2d_km = distance(Point(final_lat, final_lon), Point(tgt_lat, tgt_lon)).kilometers
        dist_3d_m = math.sqrt((dist_2d_km * 1000.0) ** 2 + (final_alt - tgt_alt) ** 2)
        flight_time = env.fdm.get_property_value("simulation/sim-time-sec")

        writer.add_scalar("episode/score", score, episode)
        writer.add_scalar("episode/steps", steps, episode)
        writer.add_scalar("episode/final_alt_m", final_alt, episode)
        writer.add_scalar("episode/distance_m", dist_3d_m, episode)
        writer.add_scalar("episode/flight_time", flight_time, episode)
        writer.add_scalar("training/entropy_coef", current_entropy_coef, episode)
        if args.lr_decay:
            writer.add_scalar("training/actor_lr", actor_opt.param_groups[0]["lr"], episode)
        if last_actor_loss is not None:
            writer.add_scalar("loss/actor", last_actor_loss, episode)
            writer.add_scalar("loss/critic", last_critic_loss, episode)
            writer.add_scalar("loss/entropy", last_entropy, episode)
        writer.add_scalar("training/buffer_size", len(rollout_buffer), episode)

        if args.save_step_data and len(step_records) > 0:
            step_dir = tb_dir / "step_data"
            step_dir.mkdir(parents=True, exist_ok=True)
            arr = np.asarray(step_records, dtype=np.float32)
            cols = np.array(
                [
                    "t_sec",
                    "reward",
                    "success",
                    "range_3d_m",
                    "ata_deg",
                    "aot_deg",
                    "act_aileron",
                    "act_elevator",
                    "act_rudder",
                    "act_throttle",
                    "tgt_act_aileron",
                    "tgt_act_elevator",
                    "tgt_act_rudder",
                    "tgt_act_throttle",
                    "agent_lat_deg",
                    "agent_lon_deg",
                    "agent_alt_m",
                    "agent_psi_deg",
                    "agent_theta_deg",
                    "agent_phi_deg",
                    "agent_u_fps",
                    "agent_v_fps",
                    "agent_w_fps",
                    "agent_p_rad_s",
                    "agent_q_rad_s",
                    "agent_r_rad_s",
                    "tgt_lat_deg",
                    "tgt_lon_deg",
                    "tgt_alt_m",
                    "tgt_psi_deg",
                    "tgt_theta_deg",
                    "tgt_phi_deg",
                    "tgt_u_fps",
                    "tgt_v_fps",
                    "tgt_w_fps",
                    "tgt_p_rad_s",
                    "tgt_q_rad_s",
                    "tgt_r_rad_s",
                ],
                dtype=object,
            )
            np.savez_compressed(step_dir / f"episode_{episode:05d}.npz", data=arr, columns=cols)

        # Elo update & stats update
        agent_score = 1.0 if mission_success else 0.0
        opp_is_anchor = isinstance(opponent, PdOpponent)
        agent_elo, opp_elo = elo_mgr.update(opponent.name, agent_score, opponent_is_anchor=opp_is_anchor)
        writer.add_scalar("elo/agent", agent_elo, episode)
        if not opp_is_anchor:
            writer.add_scalar(f"elo/opponent/{opponent.name}", opp_elo, episode)
        elo_history.append((episode, agent_elo))

        # Basic Stats
        stat = opponent_stats.setdefault(opponent.name, {"wins": 0, "games": 0})
        stat["games"] += 1
        stat["wins"] += int(agent_score)

        # -------------------------------------------------------------
        # Success-Triggered Gate Logic (P-Controller)
        # -------------------------------------------------------------
        recent_history.append(int(agent_score))
        recent_win_rate = 0.0
        if len(recent_history) > 0:
            recent_win_rate = sum(recent_history) / len(recent_history)

        triggered_update = False

        # Check trigger only when we have full window
        if len(recent_history) == trigger_window:
            target_win_rate = 0.7
            kp = 0.5
            error = recent_win_rate - target_win_rate

            # P-Control Update
            delta = kp * error
            old_ratio = self_play_ratio
            self_play_ratio = np.clip(self_play_ratio + delta, 0.0, 1.0)

            triggered_update = True
            print(f"[CONTROL] Win {recent_win_rate:.2f} (Target {target_win_rate}). Ratio {old_ratio:.2f} -> {self_play_ratio:.2f}")

            # Pool Addition Trigger (If performance is robust)
            # Since target is 0.7, we catch "good" models slightly above target
            if recent_win_rate >= 0.8:
                ckpt_path = weight_dir / f"selfplay_pool_{episode:05d}.pth"
                torch.save(
                    {
                        "actor": actor.state_dict(),
                        "critic": critic.state_dict(),
                        "actor_optimizer": actor_opt.state_dict(),
                        "critic_optimizer": critic_opt.state_dict(),
                        "agent_elo": agent_elo,
                    },
                    ckpt_path,
                )

                opp_name = f"past_self_{episode:05d}"
                past_opponents[opp_name] = PastSelfOpponent(
                    name=opp_name,
                    checkpoint_path=ckpt_path,
                    state_size=state_size,
                    device=device,
                )
                elo_mgr.set_opponent_rating(opp_name, agent_elo)
                opponent_stats.setdefault(opp_name, {"wins": 0, "games": 0})
                print(f"[POOL] Performance > 0.8. Added {opp_name} to opponent pool.")

        # Reset history if updated (to evaluate new ratio settings freshness)
        if triggered_update:
            recent_history.clear()

        writer.add_scalar("training/self_play_ratio", self_play_ratio, episode)
        writer.add_scalar("training/recent_win_rate", recent_win_rate, episode)

        # Normal periodic checkpoint (every 20 episodes)
        if episode % 20 == 0:
            ckpt_path = weight_dir / f"selfplay_epi_{episode:05d}.pth"
            torch.save(
                {
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "actor_optimizer": actor_opt.state_dict(),
                    "critic_optimizer": critic_opt.state_dict(),
                    "agent_elo": agent_elo,
                },
                ckpt_path,
            )

        action_str = (
            np.array2string(last_action.flatten(), precision=2, separator=", ")
            if last_action is not None
            else "N/A"
        )
        entropy_str = f"{last_entropy:.4f}" if last_entropy is not None else "N/A"
        success_mark = " [WIN]" if mission_success else ""

        print(
            "{} episode | score: {:.2f} | time: {:.2f} | entropy: {} | action: {} | win: {:.2f} | ratio: {:.1f} | elo: {:.1f}{}".format(
                episode, score, flight_time, entropy_str, action_str, recent_win_rate, self_play_ratio, agent_elo, success_mark
            )
        )

    writer.close()

    # Save Elo history
    if elo_history:
        elo_arr = np.asarray(elo_history, dtype=np.float32)
        elo_csv_path = tb_dir / "elo_history.csv"
        np.savetxt(elo_csv_path, elo_arr, delimiter=",", header="episode,agent_elo", comments="")
        try:
            import matplotlib.pyplot as plt  # Local import

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(elo_arr[:, 0], elo_arr[:, 1], label="Agent Elo", color="blue")
            ax.axhline(1000.0, linestyle="--", color="gray", alpha=0.7, label="PD Anchor (1000)")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Elo Rating")
            ax.set_title("Elo vs Episode")
            ax.legend()
            fig.tight_layout()
            fig.savefig(tb_dir / "elo_history.png", dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"Failure: could not save Elo history plot. Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--agent_steps", type=int, default=5)
    parser.add_argument("--settle_steps", type=int, default=5)

    # PPO hyperparameters
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--lr_decay", action=argparse.BooleanOptionalAction, default=False,
                        help="Linear LR decay (actor_lr -> min_lr). Recommended OFF for self-play (non-stationary env).")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="LR decay floor (used only when --lr_decay is set)")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--ppo_epochs", type=int, default=10, help="Number of PPO update epochs per n_steps batch")
    parser.add_argument("--n_steps", type=int, default=2048,
                        help="Number of env steps to collect before each PPO update (across episodes)")
    parser.add_argument("--mini_batch_size", type=int, default=512, help="Mini-batch size for PPO updates")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Initial entropy bonus coefficient")
    parser.add_argument("--entropy_coef_end", type=float, default=0.002,
                        help="Final entropy coef after linear decay over all episodes")
    parser.add_argument("--value_coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")

    parser.add_argument(
        "--target_action",
        type=float,
        nargs=4,
        default=[0.10355409, -0.53566162, 0.0987917, 0.37810675],
    )
    parser.add_argument("--target_offset_lat", type=float, default=0.05)
    parser.add_argument("--target_offset_long", type=float, default=0.02)
    parser.add_argument("--target_offset_alt", type=float, default=300.0)

    parser.add_argument("--pd_k_heading_to_bank", type=float, default=0.8)
    parser.add_argument("--pd_k_elev_to_pitch", type=float, default=1.0)
    parser.add_argument("--pd_k_bank_p", type=float, default=0.06)
    parser.add_argument("--pd_k_bank_d", type=float, default=0.02)
    parser.add_argument("--pd_k_pitch_p", type=float, default=0.05)
    parser.add_argument("--pd_k_pitch_d", type=float, default=0.02)
    parser.add_argument("--pd_k_rudder_damp", type=float, default=0.05)
    parser.add_argument("--pd_max_bank_deg", type=float, default=55.0)
    parser.add_argument("--pd_max_pitch_cmd_deg", type=float, default=20.0)
    parser.add_argument("--pd_aileron_sign", type=float, default=1.0)
    parser.add_argument("--pd_elevator_sign", type=float, default=-1.0)
    parser.add_argument("--pd_throttle", type=float, default=0.38)

    parser.add_argument("--init_noise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--init_noise_seed", type=int, default=None)
    parser.add_argument("--noise_target_sigma_north_m", type=float, default=500.0)
    parser.add_argument("--noise_target_sigma_east_m", type=float, default=500.0)
    parser.add_argument("--noise_target_sigma_alt_m", type=float, default=100.0)
    parser.add_argument("--noise_agent_sigma_u_fps", type=float, default=50.0)
    parser.add_argument("--noise_agent_heading_uniform_deg", type=float, default=20.0)
    parser.add_argument("--noise_target_sigma_u_fps", type=float, default=0.0)
    parser.add_argument("--noise_target_heading_uniform_deg", type=float, default=0.0)

    parser.add_argument("--elo_k_factor", type=float, default=32.0)
    parser.add_argument("--save_step_data", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--step_data_stride", type=int, default=5)

    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="20260226_PPO_Self_Play")
    args = parser.parse_args()

    train(args)
