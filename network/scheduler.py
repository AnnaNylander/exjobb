
class Scheduler(object):
    def __init__(self, param_group, max_steps=1, start_val=1, center_val=0):
        self.param_group = param_group
        self.max_steps = max_steps
        self.start_val = start_val
        self.center_val = center_val
        #self.current = start_val
        self.step_size = (center_val - start_val)/(max_steps/2)

    def update(self, optimizer, step):
        half_steps = self.max_steps // 2
        if step <= half_steps:
            current = self.start_val + step*self.step_size
        elif step > half_steps and step <= self.max_steps:
            current = self.center_val - (step - self.max_steps//2) * self.step_size
        else:
            current = max(self.start_val - (step-self.max_steps)*(self.step_size/4), self.start_val/10)
        print(self.param_group + " %.5e" %current)
        #self.current = self.start_val + min(step,max_steps//2)*self.step_size - min(step-max_steps//2,max_steps//2)*self.step_size -
        optimizer.param_groups[0][self.param_group] = current

    def serialize(self):
        dic = {}
        dic['param_group'] = self.param_group
        dic['max_steps'] = self.max_steps
        dic['start_val'] = self.start_val
        dic['center_val'] = self.center_val
        #dic['current'] = self.current
        dic['step_size'] = self.step_size
        return dic

    def deserialize(self, dic):
        self.param_group = dic['param_group']
        self.max_steps = dic['max_steps']
        self.start_val = dic['start_val']
        self.center_val = dic['center_val']
        #self.current = dic['current']
        self.step_size = dic['step_size']
