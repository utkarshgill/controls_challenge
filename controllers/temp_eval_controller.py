
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch
from controllers import BaseController
import importlib.util

train_path = Path("/Users/engelbart/Desktop/stuff/controls_challenge/experiments/exp042_simple_mlp") / "train.py"
spec = importlib.util.spec_from_file_location("exp042_train", train_path)
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)

class Controller(BaseController):
    def __init__(self):
        device = torch.device('cpu')
        self.actor = train_module.SimpleActor().to(device)
        model_path = Path("/Users/engelbart/Desktop/stuff/controls_challenge/experiments/exp042_simple_mlp") / "temp_eval_model.pth"
        self.actor.load_state_dict(torch.load(model_path, map_location=device))
        self.actor.eval()
        self.controller = train_module.SimpleController(self.actor, deterministic=True)
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        return self.controller.update(target_lataccel, current_lataccel, state, future_plan)
    
    def reset(self):
        self.controller.reset()
