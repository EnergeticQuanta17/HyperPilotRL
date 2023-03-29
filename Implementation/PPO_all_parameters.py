#####################################################################
##             LEARNING RATE SCHEDULER (learning_rate)             ##
#####################################################################

import torch.optim.lr_scheduler as lr_scheduler

def train_model(optimizer, scheduler=None):
    # training loop here
    # ...
    if scheduler:
        scheduler.step()

def step_decay(optimizer, initial_lr=0.001, drop_rate=0.1, epochs_drop=10):
    scheduler = lr_scheduler.StepLR(optimizer, step_size=epochs_drop, gamma=drop_rate)
    return scheduler

def exp_decay(optimizer, initial_lr=0.001, decay_rate=0.96):
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
    return scheduler

def cosine_anneal(optimizer, initial_lr=0.001, epochs=100):
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    return scheduler

def reduce_on_plateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False):
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, verbose=verbose)
    return scheduler

def cyclical_lr(optimizer, step_size=2000, base_lr=1e-3, max_lr=6e-3, mode='triangular', gamma=1.):
    scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size,
                                      mode=mode, gamma=gamma)
    return scheduler

def warmup_lr(optimizer, factor=10, warmup_epochs=5):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return factor
        else:
            return 1.0
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler

def one_cycle_lr(optimizer, num_steps, lr_range=(1e-4, 1e-2), momentum_range=(0.85, 0.95)):
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr_range[1], total_steps=num_steps,
                                         pct_start=0.3, anneal_strategy='cos', cycle_momentum=True,
                                         base_momentum=momentum_range[0], max_momentum=momentum_range[1])
    return scheduler

#####################################################################
##                      LEARNING RATE SCHEDULER                    ##
#####################################################################


#####################################################################
##                      LEARNING RATE SCHEDULER                    ##
#####################################################################



#####################################################################
##                      LEARNING RATE SCHEDULER                    ##
#####################################################################



#####################################################################
##                      LEARNING RATE SCHEDULER                    ##
#####################################################################



#####################################################################
##                      LEARNING RATE SCHEDULER                    ##
#####################################################################





#####################################################################
##                      LEARNING RATE SCHEDULER                    ##
#####################################################################





#####################################################################
##                      LEARNING RATE SCHEDULER                    ##
#####################################################################











all_policy = [
    "MlpPolicy",
    "CnnPolicy",
    "MultiInputPolicy",
]

# ~!@ environment is fixed {maybe we can find stronger relations between hyperparameters across other environments}
    # Very important idea

learning_rate = 
