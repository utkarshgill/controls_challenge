
from controllers import BaseController
class Controller(BaseController):
    def __init__(self):
        self.error_integral = 0.0
        self.prev_error = 0.0
        self.p = 0.1949999332
        self.i = 0.1000000164
        self.d = -0.0529999360
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        action = self.p * error + self.i * self.error_integral + self.d * error_diff
        return action
    
    def reset(self):
        self.error_integral = 0.0
        self.prev_error = 0.0
