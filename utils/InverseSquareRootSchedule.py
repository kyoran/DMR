
class InverseSquareRootSchedule(object):

    def __init__(self, warmup_step=4e4):
        if warmup_step is None:
            self.warmup_step = warmup_step
        else:
            warmup_step = int(warmup_step)
            assert warmup_step > 0 and isinstance(warmup_step, int)
            self.warmup_step = warmup_step
            init = 5e-4
            end = 1
            self.init_lr = init
            self.lr_step = (end - init) / warmup_step
            self.decay = warmup_step ** 0.5

    def step(self, step):
        if self.warmup_step is None:
            return  1
        else:
            if step < self.warmup_step:
                return self.init_lr + self.lr_step * step
            else:
                return self.decay * (step ** -0.5)