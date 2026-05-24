# Advanced Methods and Failure Modes

## 1. Curriculum Learning

Curriculum learning reduces optimization difficulty by sequencing tasks from easy to hard:

\[
\mathcal{T}_1 \rightarrow \mathcal{T}_2 \rightarrow \dots \rightarrow \mathcal{T}_n
\]

Where each task increases complexity in:

- track curvature
- traffic density
- opponent strength
- speed regime
- observation noise

Adaptive curriculum can use performance triggers:

- promote when finish rate exceeds a threshold
- demote if crash rate spikes after difficulty increases

## 2. Adversarial Racing

Adversarial racing introduces opponents whose objective is to expose the weaknesses of the main agent rather than merely drive well.

Possible adversary roles:

- blocker that forces poor overtakes
- high-speed challenger that pressures braking decisions
- disturbance agent that occupies optimal racing lines

Training forms:

- alternating self-play
- minimax-style robust RL approximations
- league training with exploiters

## 3. Imitation Learning and Hybrid Training

Pure PPO from scratch is sample inefficient for complex driving. A hybrid stack is often better:

1. expert data collection
2. behavior cloning
3. offline validation
4. online PPO fine-tuning
5. self-play or adversarial fine-tuning

This reduces early instability and helps the policy discover sane driving priors before reward optimization dominates behavior.

## 4. Transformer Driving Policies

Transformers are attractive when the scene is relational. Each vehicle and waypoint can be encoded as a token, with attention learning the interaction structure.

Potential architecture:

- entity encoder for nearby actors
- temporal stack over recent frames
- transformer backbone
- action head and value head

Useful additions:

- relative positional encoding for lane or map geometry
- masked attention over variable actor sets
- cross-attention between ego and route tokens

## 5. Failure Modes

Common system-level failure modes:

- simulator version drift changes observation semantics
- reward hacking leads to non-racing behavior
- self-play collapses into overfitting against a narrow opponent pool
- ELO inflation caused by weak matchmaking
- distributed workers run stale weights too long
- GPU learner sits idle while rollout workers bottleneck on CPU stepping
- checkpoint lineage is lost, making reproducibility impossible
- training reward rises while held-out win rate falls

Common policy-level failure modes:

- oscillatory steering
- brake-throttle chattering
- unsafe overtakes
- hugging lane boundaries to exploit reward terms
- strategic fragility on unseen maps
- collision-prone recovery behavior after perturbations

## 6. Diagnosing Reward Hacking

Symptoms:

- high episodic reward with low finish rate
- progress inflation without actual race wins
- repeated exploit loops near waypoints or lane markers

Mitigations:

- remove easy-to-game dense terms
- add success-gated reward components
- inspect trajectory videos, not just scalar dashboards
- use held-out competitive evaluation

## 7. Distributed Training Failure Analysis

Operational metrics to monitor:

- rollout samples per second
- learner queue backlog
- policy weight freshness
- pod restart count
- checkpoint write failures
- simulator crash frequency

Typical remediation:

- cap environment count per worker before CPU thrash
- pin simulator versions in container images
- separate evaluation jobs from training jobs
- use canary rollouts for infrastructure changes

## 8. Kubernetes Failure Modes

Cluster-level problems:

- GPU fragmentation from poor scheduling
- object storage throttling under heavy checkpoint traffic
- runaway hyperparameter sweeps consuming all CPU nodes
- logging saturation causing I/O stalls

Controls:

- node pool quotas
- priority classes
- per-namespace budget limits
- checkpoint retention policies
- autoscaler guardrails

## 9. Research Roadmap

Short-term:

- stronger reward shaping and evaluation coverage
- opponent pool self-play
- RLlib trainer integration

Mid-term:

- imitation learning bootstrap
- transformer policy experiments
- richer leaderboard backend and arena registry

Long-term:

- hybrid planning plus policy control
- robust adversarial leagues
- map-conditional ratings and meta-matchmaking

