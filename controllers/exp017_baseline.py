"""Baseline: 1-neuron controller"""
import torch
import torch.nn as nn
from pathlib import Path

class Controller:
    def __init__(self):
        self.net = nn.Linear(3, 1, bias=False)
        checkpoint = torch.load(
            Path(__file__).parent.parent / 'experiments/exp017_baseline/model.pth',
            map_location='cpu', weights_only=False
        )
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.net.eval()
        self.error_integral = 0.0
        self.prev_error = 0.0
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        x = torch.FloatTensor([[error, self.error_integral, error_diff]])
        
        with torch.no_grad():
            action = self.net(x).item()
        
        self.prev_error = error
        return float(max(-2.0, min(2.0, action)))



