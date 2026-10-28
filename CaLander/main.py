"""Hybrid cellular-automaton + genetic-algorithm controller for LunarLander-v3.

Examples
--------
Train:
    python main.py train --generations 30 --population 40 --episodes 3
Evaluate saved genome against the original heuristic and a random policy:
    python main.py evaluate --genome best_ca_genome.json --episodes 20
Watch the saved CA controller:
    python main.py demo --genome best_ca_genome.json
Watch the original hand-written heuristic:
    python main.py demo --controller heuristic

The CA controller uses the discrete LunarLander action space. Gymnasium's
LunarLander-v3 actions are: 0=no-op, 1=fire left engine, 2=main engine,
3=fire right engine.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Callable, Iterable

import gymnasium as gym
import numpy as np


# ---------------------------------------------------------------------------
# Original baseline heuristic
# ---------------------------------------------------------------------------
def heuristic(env, s):
    """Original hand-designed controller from the supplied file."""
    angle_targ = float(s[0]) * 0.5 + float(s[2]) * 1.0
    angle_targ = float(np.clip(angle_targ, -0.4, 0.4))
    hover_targ = 0.55 * abs(float(s[0]))

    angle_todo = (angle_targ - float(s[4])) * 0.5 - float(s[5])
    hover_todo = (hover_targ - float(s[1])) * 0.5 - float(s[3]) * 0.5

    if s[6] or s[7]:
        angle_todo = 0.0
        hover_todo = -float(s[3]) * 0.5

    if getattr(env.unwrapped, "continuous", False):
        return np.clip(np.array([hover_todo * 20 - 1, -angle_todo * 20]), -1, 1)

    action = 0
    if hover_todo > abs(angle_todo) and hover_todo > 0.05:
        action = 2
    elif angle_todo < -0.05:
        action = 3
    elif angle_todo > 0.05:
        action = 1
    return action


def demo_heuristic_lander(env, seed=None, render=False):
    """Compatibility wrapper for the original demo function."""
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    while True:
        action = heuristic(env, obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        if render:
            still_open = env.render()
            if still_open is False:
                break
        if steps % 20 == 0 or terminated or truncated:
            print("observations:", " ".join(f"{x:+0.2f}" for x in obs))
            print(f"step {steps} total_reward {total_reward:+0.2f}")
        steps += 1
        if terminated or truncated:
            break
    if render:
        env.close()
    return total_reward


# ---------------------------------------------------------------------------
# Cellular automaton policy
# ---------------------------------------------------------------------------
N_CELLS = 16
N_ACTIONS = 4
N_OBS = 8
# Approximate scales for LunarLander's continuous observation features.
OBS_SCALES = np.array([1.5, 1.5, 1.5, 1.5, np.pi, 5.0], dtype=np.float32)


@dataclass
class Genome:
    """An elementary CA rule plus a linear action decoder."""
    rule: int                 # 8-bit elementary CA rule, 0..255
    weights: np.ndarray       # shape (4, N_CELLS + 1); last column is bias

    def to_dict(self) -> dict:
        return {"rule": int(self.rule), "weights": self.weights.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> "Genome":
        weights = np.asarray(data["weights"], dtype=np.float32)
        expected = (N_ACTIONS, N_CELLS + 1)
        if weights.shape != expected:
            raise ValueError(f"Genome weights must have shape {expected}, got {weights.shape}")
        rule = int(data["rule"])
        if not 0 <= rule <= 255:
            raise ValueError("Genome rule must be an integer in [0, 255]")
        return cls(rule=rule, weights=weights)


def encode_observation(obs: Iterable[float]) -> np.ndarray:
    """Encode 8 observations into 16 binary cells.

    Each of the first six continuous features becomes a sign bit and a
    magnitude bit. The last two cells encode leg contact. The final two
    cells are constant context cells, giving the CA a fixed-width ring.
    """
    s = np.asarray(obs, dtype=np.float32)
    if s.shape != (N_OBS,):
        raise ValueError(f"Expected observation shape ({N_OBS},), got {s.shape}")
    cells: list[int] = []
    normalized = np.clip(s[:6] / OBS_SCALES, -2.0, 2.0)
    for value in normalized:
        cells.append(int(value >= 0.0))
        cells.append(int(abs(float(value)) >= 0.35))
    cells.extend([int(s[6] >= 0.5), int(s[7] >= 0.5)])
    cells.extend([1, 0])
    return np.asarray(cells, dtype=np.uint8)


def ca_step(cells: np.ndarray, rule: int) -> np.ndarray:
    """Apply one synchronous radius-1 elementary CA update on a ring."""
    left = np.roll(cells, 1).astype(np.uint8)
    right = np.roll(cells, -1).astype(np.uint8)
    pattern = (left << 2) | (cells << 1) | right
    return np.asarray([(rule >> int(p)) & 1 for p in pattern], dtype=np.uint8)


def ca_policy(obs: Iterable[float], genome: Genome, iterations: int = 5) -> int:
    """Return one of the four discrete LunarLander actions."""
    cells = encode_observation(obs)
    for _ in range(iterations):
        cells = ca_step(cells, genome.rule)
    features = np.concatenate([cells.astype(np.float32), np.ones(1, dtype=np.float32)])
    scores = genome.weights @ features
    return int(np.argmax(scores))


# ---------------------------------------------------------------------------
# Genetic algorithm
# ---------------------------------------------------------------------------
def random_genome(rng: np.random.Generator) -> Genome:
    return Genome(
        rule=int(rng.integers(0, 256)),
        weights=rng.normal(0.0, 0.45, size=(N_ACTIONS, N_CELLS + 1)).astype(np.float32),
    )


def crossover(a: Genome, b: Genome, rng: np.random.Generator) -> Genome:
    # Mix the elementary CA rule bit-by-bit, and the decoder weight-by-weight.
    mask = int(rng.integers(0, 256))
    rule = (a.rule & mask) | (b.rule & (255 ^ mask))
    choose_a = rng.random(a.weights.shape) < 0.5
    weights = np.where(choose_a, a.weights, b.weights).astype(np.float32)
    return Genome(rule=rule, weights=weights)


def mutate(genome: Genome, rng: np.random.Generator,
           bit_mutation: float = 0.04, weight_mutation: float = 0.08,
           sigma: float = 0.18) -> Genome:
    rule = int(genome.rule)
    for bit in range(8):
        if rng.random() < bit_mutation:
            rule ^= 1 << bit
    weights = genome.weights.copy()
    mask = rng.random(weights.shape) < weight_mutation
    weights[mask] += rng.normal(0.0, sigma, size=int(mask.sum())).astype(np.float32)
    np.clip(weights, -4.0, 4.0, out=weights)
    return Genome(rule=rule, weights=weights)


def tournament_select(population: list[Genome], fitness: np.ndarray,
                      rng: np.random.Generator, tournament_size: int = 3) -> Genome:
    candidates = rng.integers(0, len(population), size=tournament_size)
    winner = int(candidates[np.argmax(fitness[candidates])])
    return population[winner]


def rollout(env, controller: Callable, seed: int, render: bool = False) -> tuple[float, int, bool]:
    obs, _ = env.reset(seed=int(seed))
    total_reward = 0.0
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        action = controller(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        steps += 1
        if render:
            env.render()
    # LunarLander's conventional solved threshold is 200 mean reward.
    return total_reward, steps, total_reward >= 200.0


def evaluate_genome(genome: Genome, seeds: Iterable[int], env_id: str = "LunarLander-v3",
                    iterations: int = 5) -> dict:
    env = gym.make(env_id)
    rewards, lengths, successes = [], [], []
    try:
        for seed in seeds:
            reward, length, success = rollout(
                env, lambda obs: ca_policy(obs, genome, iterations), int(seed)
            )
            rewards.append(reward)
            lengths.append(length)
            successes.append(success)
    finally:
        env.close()
    return _summarize(rewards, lengths, successes)


def evaluate_controller(controller: Callable, seeds: Iterable[int], env_id: str = "LunarLander-v3") -> dict:
    env = gym.make(env_id)
    rewards, lengths, successes = [], [], []
    try:
        for seed in seeds:
            reward, length, success = rollout(env, controller, int(seed))
            rewards.append(reward)
            lengths.append(length)
            successes.append(success)
    finally:
        env.close()
    return _summarize(rewards, lengths, successes)


def _summarize(rewards, lengths, successes) -> dict:
    rewards = np.asarray(rewards, dtype=np.float64)
    return {
        "episodes": int(len(rewards)),
        "mean_reward": float(np.mean(rewards)) if len(rewards) else float("nan"),
        "std_reward": float(np.std(rewards)) if len(rewards) else float("nan"),
        "median_reward": float(np.median(rewards)) if len(rewards) else float("nan"),
        "success_rate": float(np.mean(successes)) if len(successes) else float("nan"),
        "mean_steps": float(np.mean(lengths)) if len(lengths) else float("nan"),
        "rewards": [float(x) for x in rewards],
    }


def train(args) -> Genome:
    rng = np.random.default_rng(args.seed)
    population = [random_genome(rng) for _ in range(args.population)]
    # Fixed seeds make candidate comparisons less noisy; validation seeds are
    # kept separate and are only used after training.
    train_seeds = [args.seed + 1000 + i for i in range(args.episodes)]
    history = []
    best_genome = None
    best_fitness = -np.inf

    env = gym.make(args.env_id)
    try:
        for generation in range(args.generations):
            fitness = np.empty(len(population), dtype=np.float64)
            for i, genome in enumerate(population):
                episode_rewards = []
                for episode_seed in train_seeds:
                    reward, _, _ = rollout(
                        env,
                        lambda obs, g=genome: ca_policy(obs, g, args.iterations),
                        episode_seed,
                    )
                    episode_rewards.append(reward)
                fitness[i] = float(np.mean(episode_rewards))

            order = np.argsort(fitness)[::-1]
            gen_best = float(fitness[order[0]])
            gen_mean = float(np.mean(fitness))
            if gen_best > best_fitness:
                best_fitness = gen_best
                best_genome = Genome(population[order[0]].rule, population[order[0]].weights.copy())
                save_genome(best_genome, args.genome, metadata={
                    "generation": generation,
                    "training_fitness": best_fitness,
                    "seed": args.seed,
                    "iterations": args.iterations,
                })
            history.append({"generation": generation, "best": gen_best, "mean": gen_mean})
            print(f"Generation {generation + 1:03d}/{args.generations}: "
                  f"best={gen_best:8.2f} mean={gen_mean:8.2f} "
                  f"all-time-best={best_fitness:8.2f}")

            # Elitism: keep the strongest candidates unchanged.
            elite_count = max(1, min(args.elites, args.population))
            next_population = [
                Genome(population[i].rule, population[i].weights.copy())
                for i in order[:elite_count]
            ]
            while len(next_population) < args.population:
                parent_a = tournament_select(population, fitness, rng)
                parent_b = tournament_select(population, fitness, rng)
                child = crossover(parent_a, parent_b, rng)
                child = mutate(child, rng, args.bit_mutation, args.weight_mutation, args.mutation_sigma)
                next_population.append(child)
            population = next_population
    finally:
        env.close()

    with open(args.history, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    assert best_genome is not None
    print(f"\nSaved best genome to: {args.genome}")
    print(f"Best training fitness: {best_fitness:.2f}")

    validation_seeds = [args.seed + 100_000 + i for i in range(args.validation_episodes)]
    print("\nHeld-out validation (not used during evolution):")
    print_results("CA + GA", evaluate_genome(best_genome, validation_seeds, args.env_id, args.iterations))
    print_results("Original heuristic", evaluate_controller(
        lambda obs: heuristic_proxy(args.env_id, obs), validation_seeds, args.env_id
    ))
    print_results("Random policy", evaluate_random(validation_seeds, args.env_id, seed=args.seed + 200_000))
    return best_genome


def heuristic_proxy(env_id: str, obs: np.ndarray) -> int:
    """Call the original heuristic with a discrete LunarLander-compatible env."""
    # The heuristic only checks env.unwrapped.continuous. This small proxy
    # avoids constructing an environment at every policy step.
    class _Unwrapped:
        continuous = False
    class _Env:
        unwrapped = _Unwrapped()
    return int(heuristic(_Env(), obs))


def save_genome(genome: Genome, path: str, metadata: dict | None = None) -> None:
    payload = {"format_version": 1, "genome": genome.to_dict(), "metadata": metadata or {}}
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(temporary_path, path)


def load_genome(path: str) -> tuple[Genome, dict]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    # Also accept a bare Genome dictionary for convenience.
    if "genome" in payload:
        return Genome.from_dict(payload["genome"]), payload.get("metadata", {})
    return Genome.from_dict(payload), {}


def print_results(name: str, result: dict) -> None:
    print(f"{name:20s} mean={result['mean_reward']:8.2f} "
          f"std={result['std_reward']:7.2f} median={result['median_reward']:8.2f} "
          f"success@200={100 * result['success_rate']:5.1f}% "
          f"mean_steps={result['mean_steps']:6.1f} (n={result['episodes']})")


def run_evaluation(args) -> None:
    genome, metadata = load_genome(args.genome)
    iterations = args.iterations if args.iterations is not None else int(metadata.get("iterations", 5))
    seeds = [args.seed + i for i in range(args.episodes)]
    print(f"Evaluation seeds: {seeds[0]}..{seeds[-1]}")
    print_results("CA + GA", evaluate_genome(genome, seeds, args.env_id, iterations))
    print_results("Original heuristic", evaluate_controller(
        lambda obs: heuristic_proxy(args.env_id, obs), seeds, args.env_id
    ))
    # Use deterministic, episode-specific random policies for repeatability.
    random_results = evaluate_random(seeds, args.env_id, seed=args.seed + 300_000)
    print_results("Random policy", random_results)


def evaluate_random(seeds: Iterable[int], env_id: str, seed: int = 0) -> dict:
    env = gym.make(env_id)
    rng = np.random.default_rng(seed)
    rewards, lengths, successes = [], [], []
    try:
        for episode_index, episode_seed in enumerate(seeds):
            obs, _ = env.reset(seed=int(episode_seed))
            # Seed the environment's action space too, for reproducibility.
            env.action_space.seed(seed + episode_index)
            total_reward = 0.0
            steps = 0
            terminated = truncated = False
            while not (terminated or truncated):
                action = int(rng.integers(0, N_ACTIONS))
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                steps += 1
            rewards.append(total_reward)
            lengths.append(steps)
            successes.append(total_reward >= 200.0)
    finally:
        env.close()
    return _summarize(rewards, lengths, successes)


def run_demo(args) -> None:
    render_mode = "human"
    env = gym.make(args.env_id, render_mode=render_mode)
    try:
        if args.controller == "heuristic":
            demo_heuristic_lander(env, seed=args.seed, render=True)
        else:
            genome, metadata = load_genome(args.genome)
            iterations = args.iterations if args.iterations is not None else int(metadata.get("iterations", 5))
            reward, steps, success = rollout(
                env, lambda obs: ca_policy(obs, genome, iterations), args.seed, render=True
            )
            print(f"CA controller: reward={reward:.2f}, steps={steps}, reward>=200={success}")
    finally:
        # demo_heuristic_lander closes its env; close is safe to repeat.
        env.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="evolve and save a CA controller")
    p_train.add_argument("--env-id", default="LunarLander-v3")
    p_train.add_argument("--population", type=int, default=40)
    p_train.add_argument("--generations", type=int, default=30)
    p_train.add_argument("--episodes", type=int, default=3, help="training episodes per genome per generation")
    p_train.add_argument("--validation-episodes", type=int, default=10)
    p_train.add_argument("--iterations", type=int, default=5, help="CA updates per environment step")
    p_train.add_argument("--seed", type=int, default=7)
    p_train.add_argument("--elites", type=int, default=2)
    p_train.add_argument("--bit-mutation", type=float, default=0.04)
    p_train.add_argument("--weight-mutation", type=float, default=0.08)
    p_train.add_argument("--mutation-sigma", type=float, default=0.18)
    p_train.add_argument("--genome", default="best_ca_genome.json")
    p_train.add_argument("--history", default="ca_ga_history.json")

    p_eval = sub.add_parser("evaluate", help="compare a saved genome with baselines")
    p_eval.add_argument("--env-id", default="LunarLander-v3")
    p_eval.add_argument("--genome", default="best_ca_genome.json")
    p_eval.add_argument("--episodes", type=int, default=20)
    p_eval.add_argument("--iterations", type=int, default=None)
    p_eval.add_argument("--seed", type=int, default=50_000)

    p_demo = sub.add_parser("demo", help="render one episode")
    p_demo.add_argument("--env-id", default="LunarLander-v3")
    p_demo.add_argument("--controller", choices=["ca", "heuristic"], default="ca")
    p_demo.add_argument("--genome", default="best_ca_genome.json")
    p_demo.add_argument("--iterations", type=int, default=None)
    p_demo.add_argument("--seed", type=int, default=123)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        if args.population < 2 or args.generations < 1 or args.episodes < 1:
            parser.error("population must be >=2, generations and episodes must be >=1")
        train(args)
    elif args.command == "evaluate":
        if args.episodes < 1:
            parser.error("episodes must be >=1")
        run_evaluation(args)
    elif args.command == "demo":
        run_demo(args)


if __name__ == "__main__":
    main()
