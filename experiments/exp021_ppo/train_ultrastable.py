"""
PPO with ultra-low noise + lots of data
σ=0.001 (minimal exploration) + 100 routes/epoch (stable gradients)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, CONTROL_START_IDX, COST_END_IDX, DEL_T, LAT_ACCEL_COST_MULTIPLIER

state_dim, action_dim, hidden_dim = 53, 1, 128
max_epochs, routes_per_epoch, eval_interval = 50, 100, 2  # 100 routes!
batch_size, K_epochs, lr = 4000, 10, 3e-4
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
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 7.0)  # σ ≈ 0.0009
    
    def forward(self, s):
        return self.actor(s), self.log_std.exp(), self.critic(s)

class Ctrl:
    def __init__(self, ac, collect=False):
        self.ac, self.collect = ac, collect
        self.ei, self.pe = 0.0, 0.0
        self.states, self.actions, self.rewards = [], [], []
        self.pl, self.sc = None, 0
    
    def update(self, target, current, state, future_plan):
        e = target - current
        self.ei += e
        ed = e - self.pe
        self.pe = e
        curvs = [(future_plan.lataccel[i] - state.roll_lataccel) / max(state.v_ego**2, 1.0) 
                 if i < len(future_plan.lataccel) else 0.0 for i in range(49)]
        raw_state = np.array([e, self.ei, ed, state.v_ego] + curvs, dtype=np.float32)
        
        if self.collect:
            s_t = torch.as_tensor(raw_state / OBS_SCALE, dtype=torch.float32).to(device)
            with torch.no_grad():
                mean, std, _ = self.ac(s_t)
                a = torch.distributions.Normal(mean, std).sample()
            
            lat_err = (target - current) ** 2 * 100
            r = 0.0
            if self.pl and self.sc >= CONTROL_START_IDX and self.sc < COST_END_IDX:
                jerk = ((current - self.pl) / DEL_T) ** 2 * 100
                r = -(lat_err * LAT_ACCEL_COST_MULTIPLIER + jerk) / 1000.0
            
            self.states.append(s_t)
            self.actions.append(a)
            self.rewards.append(r)
            self.pl = current
            self.sc += 1
            return float(np.clip(a.item(), -2.0, 2.0))
        else:
            with torch.no_grad():
                mean = self.ac.actor(torch.FloatTensor(raw_state / OBS_SCALE))
            return float(np.clip(mean.item(), -2.0, 2.0))

def compute_advantages_per_episode(rewards, values, gamma, lam):
    T = len(rewards)
    advs, gae = [], 0.0
    vals_list = values.squeeze().tolist() + [0.0]
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * vals_list[t+1] - vals_list[t]
        gae = delta + gamma * lam * gae
        advs.insert(0, gae)
    advs = torch.FloatTensor(advs).to(device)
    rets = advs + torch.FloatTensor(vals_list[:-1]).to(device)
    return advs, rets

class PPO:
    def __init__(self, ac):
        self.ac = ac
        self.opt = optim.Adam(ac.parameters(), lr=lr)
    
    def update(self, ctrls):
        all_states, all_actions, all_old_logprobs, all_advs, all_rets = [], [], [], [], []
        
        for ctrl in ctrls:
            if len(ctrl.states) == 0:
                continue
            states = torch.stack(ctrl.states)
            actions = torch.stack(ctrl.actions)
            
            with torch.no_grad():
                means, stds, values = self.ac(states)
                dist = torch.distributions.Normal(means, stds)
                old_logprobs = dist.log_prob(actions).sum(-1)
                advs, rets = compute_advantages_per_episode(ctrl.rewards, values, gamma, gae_lambda)
            
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

def rollout(ac, f, collect=False):
    ctrl = Ctrl(ac, collect=collect)
    sim = TinyPhysicsSimulator(model_onnx, str(f), controller=ctrl)
    cost = sim.rollout()['total_cost']
    return ctrl if collect else None, cost

def evaluate(ac, files, n=10):
    return np.mean([rollout(ac, f, False)[1] for f in files[:n]])

def train():
    ac = ActorCritic().to(device)
    bc = torch.load('./experiments/exp020_normalized/model.pth', map_location='cpu', weights_only=False)
    ac.actor.load_state_dict(bc['model_state_dict'])
    
    bc_cost = evaluate(ac, test_files)
    print(f"BC init: {bc_cost:.2f}, σ={ac.log_std.exp().item():.6f}")
    print("Strategy: ULTRA-LOW noise + HIGH data (100 routes/epoch)\n")
    
    ppo = PPO(ac)
    best = bc_cost
    
    for ep in trange(max_epochs):
        ctrls, costs = [], []
        for f in random.sample(train_files, routes_per_epoch):
            c, cost = rollout(ac, f, collect=True)
            ctrls.append(c)
            costs.append(cost)
        
        loss = ppo.update(ctrls)
        
        if ep % eval_interval == 0:
            test_cost = evaluate(ac, test_files, 10)
            if test_cost < best:
                best = test_cost
                torch.save({'model_state_dict': ac.state_dict(), 'obs_scale': OBS_SCALE}, 
                           'experiments/exp021_ppo/model_ultrastable.pth')
            print(f"E{ep:2d} tr={np.mean(costs):5.1f} tst={test_cost:5.1f} best={best:5.1f} σ={ac.log_std.exp().item():.6f}", flush=True)
    
    print(f"\n✅ Best: {best:.2f} (BC: {bc_cost:.2f}, Δ={bc_cost-best:.2f})")

if __name__ == '__main__':
    train()



