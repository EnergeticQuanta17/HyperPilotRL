LearningRate class
    [Copied] Step Decay: In this scheduler, the learning rate is decreased by a factor after a fixed number of epochs.

    [Copied] Exponential Decay: In this scheduler, the learning rate is exponentially decreased after each epoch.

    [Copied]Cosine Annealing: In this scheduler, the learning rate is decreased according to a cosine function.

    [Copied] Reduce LR on Plateau: In this scheduler, the learning rate is reduced when the validation loss plateaus.

    Cyclical Learning Rates: In this scheduler, the learning rate is increased and decreased in a cyclic manner, which helps to escape from local minima.

    Learning Rate Warmup: In this scheduler, the learning rate is slowly increased from a small value to the optimal value in the beginning of training.

    One-Cycle Learning Rate: In this scheduler, the learning rate is increased to a maximum value, then decreased to a minimum value, and then increased again to the optimal value.


    Put the learning_rate schedulers in a class

    