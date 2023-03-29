import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K

class StepDecay(LearningRateSchedule):
    def __init__(self, initial_lr, drop_factor, epochs_drop):
        super(StepDecay, self).__init__()
        self.initial_lr = initial_lr
        self.drop_factor = drop_factor
        self.epochs_drop = epochs_drop

    def __call__(self, epoch):
        lr = self.initial_lr * (self.drop_factor ** (epoch // self.epochs_drop))
        return lr

class ExpDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, decay_rate, decay_steps):
        super(ExpDecay, self).__init__()
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        
    def __call__(self, step):
        lr = self.initial_lr * tf.math.pow(self.decay_rate, tf.math.floor(step / self.decay_steps))
        return lr

class CosineAnnealingScheduler(Callback):
    def __init__(self, max_lr, min_lr, T_0, T_mult=1, verbose=0):
        super(CosineAnnealingScheduler, self).__init__()
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.T_0 = T_0
        self.T_mult = T_mult
        self.verbose = verbose
        self.cycle_counter = 0
        self.epoch_counter = 0
        self.lr = K.get_value(self.model.optimizer.lr)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_counter += 1
        if self.epoch_counter > self.T_0:
            self.epoch_counter = 0
            self.T_0 *= self.T_mult
            self.cycle_counter += 1
        lr = self.min_lr + 0.5 * (

import tensorflow as tf

class ReduceLRonPlateauScheduler(tf.keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', factor=0.1, patience=10, verbose=0,
                 mode='auto', min_delta=1e-4, cooldown=0, min_lr=0, **kwargs):
        super(ReduceLRonPlateauScheduler, self).__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.min_delta = min_delta
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.best = None
        self.wait = 0
        self._reset()
        self._cooldown_counter = 0
        self.lr_history = []
        
    def _reset(self):
        if self.mode == 'min' or self.mode == 'auto' and 'acc' not in self.monitor:
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.wait = 0
        self.best_lr = None
        
    def on_train_begin(self, logs=None):
        self._reset()
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = self.model.optimizer.lr.numpy()
        self.lr_history.append(logs['lr'])
        
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(f'Reduce LR on Plateau conditioned on metric `{self.monitor}` '
                          'which is not available. Available metrics are: ' + ','.join(list(logs.keys())),
                          RuntimeWarning)
            return

        if self.in_cooldown():
            self._cooldown_counter -= 1
            self.wait = 0

        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
            self.best_lr = self.model.optimizer.lr.numpy()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = float(self.model.optimizer.lr.numpy())
                if old_lr > self.min_lr:
                    new_lr = old_lr * self.factor
                    new_lr = max(new_lr, self.min_lr)
                    if self.verbose > 0:
                        print(f'\nEpoch {epoch + 1}: Reduce LR on Plateau reducing learning rate '
                              'to {new_lr:.8f}.')
                    self.model.optimizer.lr.assign(new_lr)
                    self.wait = 0
                    self._cooldown_counter = self.cooldown
                    
    def in_cooldown(self):
        return self._cooldown_counter > 0
    
    def on_train_end(self, logs=None):
        if self.verbose > 0:
            print(f'Best learning rate found: {self.best_lr:.8f}.')

class CyclicalLearningRate(tf.keras.callbacks.Callback):
    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular', gamma=1.):
        super(CyclicalLearningRate, self).__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        self.cycle = 0
        
    def on_train_begin(self, logs={}):
        logs = logs or {}
        if self.mode == 'triangular':
            self.mode_fn = lambda x: 1.
        elif self.mode == 'triangular2':
            self.mode_fn = lambda x: 1/(2.**(x-1))
        elif self.mode == 'exp_range':
            self.mode_fn = lambda x: self.gamma**(x)
        
        self.max_steps = self.params['epochs'] * self.params['steps']
        
    def on_batch_begin(self, batch, logs={}):
        logs = logs or {}
        self.cycle = np.floor(1 + batch / (2 * self.step_size))
        x = np.abs(batch / self.step_size - 2 * self.cycle + 1)
        lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.mode_fn(self.cycle)
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        
    def on_epoch_end(self, epoch, logs={}):
        logs = logs or {}
        if self.mode == 'exp_range':
            tf.keras.backend.set_value(self.model.optimizer.lr, self.base_lr * self.gamma**(epoch))



all_policy = [
    "MlpPolicy",
    "CnnPolicy",
    "MultiInputPolicy",
]

# ~!@ environment is fixed {maybe we can find stronger relations between hyperparameters across other environments}
    # Very important idea

learning_rate = 
