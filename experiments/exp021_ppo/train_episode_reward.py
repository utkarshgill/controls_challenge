"""
PPO with EPISODE reward instead of per-step reward
All steps in trajectory get same credit based on final cost
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator

state_dim, action_dim, hidden_dim = 53, 1, 128
max_epochs, routes_per_epoch, eval_interval = 100, 30, 2
batch_size, K_epochs, lr = 2000, 10, 3e-4
gamma, gae_lambda, eps_clip, entropy_coef = 0.99, 0.95, 0.2, 0.01
device = torch.device('cpu')

OBS_SCALE = np.array([0.3664, 7.1769, 0.1396, 38.7333] + [0.1573] * 49, dtype=np.float32)

all_files = sorted(Path('./data').glob('*.csv'))
random.seed(42)
random.shuffle(all_files)
train_files, val_files, test_files = all_files[:15000], all_files[15000:17500], all_files[17500:20000]

model_onnx = TinyPhysicsModel('./models/tinyphysics.onnx', debug=False)

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Tanh(), 
                                   nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, action_dim))
        self.critic = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Tanh(), 
                                    nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 5.0)
    
    def forward(self, s):
        return self.actor(s), self.log_std.exp(), self.critic(s)

class Ctrl:
    def __init__(self, ac):
        self.ac = ac
        self.ei, self.pe = 0.0, 0.0
        self.states, self.actions = [], []
    
    def update(self, target, current, state, future_plan):
        e = target - current
        self.ei += e
        ed = e - self.pe
        self.pe = e
        curvs = [(future_plan.lataccel[i] - state.roll_lataccel) / max(state.v_ego**2, 1.0) 
                 if i < len(future_plan.lataccel) else 0.0 for i in range(49)]
        raw_state = np.array([e, self.ei, ed, state.v_ego] + curvs, dtype=np.float32)
        
        s_t = torch.as_tensor(raw_state / OBS_SCALE, dtype=torch.float32).to(device)
        with torch.no_grad():
            mean, std, _ = self.ac(s_t)
            a = torch.distributions.Normal(mean, std).sample()
        
        self.states.append(s_t)
        self.actions.append(a)
        
        return float(np.clip(a.item(), -2.0, 2.0))

class EvalCtrl:
    def __init__(self, ac):
        self.ac = ac
        self.ei, self.pe = 0.0, 0.0
    
    def update(self, target, current, state, future_plan):
        e = target - current
        self.ei += e
        ed = e - self.pe
        self.pe = e
        curvs = [(future_plan.lataccel[i] - state.roll_lataccel) / max(state.v_ego**2, 1.0) 
                 if i < len(future_plan.lataccel) else 0.0 for i in range(49)]
        raw_state = np.array([e, self.ei, ed, state.v_ego] + curvs, dtype=np.float32)
        
        with torch.no_grad():
            mean = self.ac.actor(torch.FloatTensor(raw_state / OBS_SCALE))
        return float(np.clip(mean.item(), -2.0, 2.0))

def compute_advantages_per_episode(episode_reward, values):
    """
    Episode reward: single scalar for whole episode
    All steps get same reward (proportional to time remaining)
    """
    T = len(values)
    # Constant reward per step
    reward_per_step = episode_reward / T
    
    advs = []
    vals_list = values.squeeze().tolist() + [0.0]
    
    for t in range(T):
        # Advantage = actual return - predicted value
        # Actual return = reward + 0 (no future, episode ends)
        adv = reward_per_step - vals_list[t]
        advs.append(adv)
    
    advs = torch.FloatTensor(advs).to(device)
    # Returns = what we actually got (constant episode reward)
    rets = torch.full_like(advs, reward_per_step)
    
    return advs, rets

class PPO:
    def __init__(self, ac):
        self.ac = ac
        self.opt = optim.Adam(ac.parameters(), lr=lr)
    
    def update(self, episodes):
        # episodes = list of (ctrl, cost) tuples
        all_states, all_actions, all_old_logprobs, all_advs, all_rets = [], [], [], [], []
        
        for ctrl, cost in episodes:
            if len(ctrl.states) == 0:
                continue
            
            # Episode reward: negative cost (we want to minimize cost)
            episode_reward = -cost / 100.0  # Scale to reasonable range
            
            states = torch.stack(ctrl.states)
            actions = torch.stack(ctrl.actions)
            
            with torch.no_grad():
                means, stds, values = self.ac(states)
                dist = torch.distributions.Normal(means, stds)
                old_logprobs = dist.log_prob(actions).sum(-1)
                
                # Compute advantages with episode reward
                advs, rets = compute_advantages_per_episode(episode_reward, values)
            
            all_states.append(states)
            all_actions.append(actions)
            all_old_logprobs.append(old_logprobs)
            all_advs.append(advs)
            all_rets.append(rets)
        
        if len(all_states) == 0:
            return 0
        
        states = torch.cat(all_states)
        actions = torch.cat(all_actions)
        old_logprobs = torch.cat(all_old_logprobs)
        advs = torch.cat(all_advs)
        rets = torch.cat(all_rets)
        
        # Normalize advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        
        dataset = TensorDataset(states, actions, old_logprobs, advs, rets)
        
        total_loss, n = 0, 0
        for _ in range(K_epochs):
            for batch in DataLoader(dataset, batch_size=min(batch_size, len(states)), shuffle=True):
                bs, ba, bolp, badv, bret = batch
                means, stds, values = self.ac(bs)
                dist = torch.distributions.Normal(means, stds)
                logp = dist.log_prob(ba).sum(-1)
                ratios = torch.exp(logp - bolp)
                aloss = -torch.min(ratios * badv, torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * badv).mean()
                closs = F.mse_loss(values.squeeze(-1), bret)
                ent = dist.entropy().sum(-1).mean()
                loss = aloss + 0.5 * closs - entropy_coef * ent
                
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac.parameters(), max_norm=0.5)
                self.opt.step()
                
                total_loss += loss.item()
                n += 1
        
        return total_loss/n

def rollout(ac, f, train=False):
    if train:
        ctrl = Ctrl(ac)
        sim = TinyPhysicsSimulator(model_onnx, str(f), controller=ctrl)
        cost = sim.rollout()['total_cost']
        return ctrl, cost
    else:
        ctrl = EvalCtrl(ac)
        sim = TinyPhysicsSimulator(model_onnx, str(f), controller=ctrl)
        return None, sim.rollout()['total_cost']

def evaluate(ac, files, n=10):
    return np.mean([rollout(ac, f, False)[1] for f in files[:n]])

def train():
    ac = ActorCritic().to(device)
    bc = torch.load('./experiments/exp020_normalized/model.pth', map_location='cpu', weights_only=False)
    ac.actor.load_state_dict(bc['model_state_dict'])
    
    bc_cost = evaluate(ac, test_files)
    print(f"BC init: {bc_cost:.2f}, σ={ac.log_std.exp().item():.4f}")
    print("Using EPISODE-LEVEL rewards (all steps get credit for final cost)\n")
    
    ppo = PPO(ac)
    best = bc_cost
    
    for ep in trange(max_epochs):
        episodes = []
        costs = []
        for f in random.sample(train_files, routes_per_epoch):
            ctrl, cost = rollout(ac, f, train=True)
            episodes.append((ctrl, cost))
            costs.append(cost)
        
        loss = ppo.update(episodes)
        
        if ep % eval_interval == 0:
            test_cost = evaluate(ac, test_files, 10)
            if test_cost < best:
                best = test_cost
                torch.save({'model_state_dict': ac.state_dict(), 'obs_scale': OBS_SCALE}, 
                           'experiments/exp021_ppo/model_episode.pth')
            print(f"E{ep:2d} tr={np.mean(costs):5.1f} tst={test_cost:5.1f} best={best:5.1f} σ={ac.log_std.exp().item():.4f}", flush=True)
    
    print(f"\n✅ Best: {best:.2f} (BC: {bc_cost:.2f}, Δ={bc_cost-best:.2f})")

if __name__ == '__main__':
    train()



