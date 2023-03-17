
import matplotlib.pyplot as plt


class logger:
    def __init__(self):
        self.loss_record = score(
            path='./outputs/ppo_train_loss.png',
            xlabel='epochs',
            ylabel='ppo_train_loss'
        )
        self.reward_record = score(
            path='./outputs/rl_reward.png',
            xlabel='epochs',
            ylabel='reward'
        )
        self.beta_record = score(
            path='./outputs/ppo_beta.png',
            xlabel='epochs',
            ylabel='ppo_beta'
        )
        self.ce_record = score(
            path='./outputs/ce_loss.png',
            xlabel='epochs',
            ylabel='ppo_beta'
        )
        self.kl_record = score(
            path='./outputs/kl.png',
            xlabel='epochs',
            ylabel='kl'
        )
        self.names = {
            'loss': self.loss_record,
            'reward': self.reward_record,
            'beta': self.beta_record,
            'ce_loss': self.ce_record,
            'kl': self.kl_record,
        }

    def update(self, **kwargs):
        for score_name in kwargs:
            if score_name not in self.names:
                continue
            self.names[score_name].update(kwargs[score_name])


class score:
    def __init__(self, **kwargs):
        self.arr = []
        self.path = kwargs['path']
        self.xlabel = kwargs['xlabel']
        self.ylabel = kwargs['ylabel']

    def update(self, val):
        self.arr.append(val)
        self.save()

    def save(self):
        plt.plot(self.arr)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.savefig(self.path, dpi=300)
        plt.clf()
