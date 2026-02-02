"""Simple 1-neuron controller"""
from . import BaseController
import numpy as np
import torch
from pathlib import Path

class Controller(BaseController):
    def __init__(self):
        checkpoint = torch.load(
            Path(__file__).parent.parent / 'experiments/exp016_one_neuron/simple.pth',
            map_location='cpu', weights_only=False
        )
        self.net = torch.nn.Linear(3, 1, bias=False)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.net.eval()
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        s = torch.FloatTensor([error, error * state.v_ego, state.v_ego / 30.0])
        
        with torch.no_grad():
            action = self.net(s).item()
        
        return float(np.clip(action, -2.0, 2.0))



