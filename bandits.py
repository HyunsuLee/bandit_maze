# bandits.py
import numpy as np
import pymc3 as pm


class GaussianBandit():
    def __init__(self, k, mu=0, sigma=1):
        self.k = k
        self.mu = mu
        self.sigma = sigma
        self.reset()

    def reset(self):
        '''reset k Gaussian Bandits randomly with mean(mu) and SD(sigma)
        each mean of Gaussain Bandit assign to the action value(true mean)
        optimal indicate ideal Bandit
        '''
        self.action_values = np.random.normal(self.mu, self.sigma, self.k)
        self.optimal = np.argmax(self.action_values)

    def pull(self, action):
        return (np.random.normal(self.action_values[action]),
                action == self.optimal)


class BinomialBandit():
    """
    The Binomial distribution models the probability of an event occurring with
    p probability k times over N trials i.e. get heads on a p-coin k times on
    N flips.

    In the bandit scenario, this can be used to approximate a discrete user
    rating or "strength" of response to a single event.
    """

    def __init__(self, k, n, p=None, t=None):
        self.k = k
        self.n = n
        self.p = p
        self.t = t
        self.model = pm.Model()
        with self.model:
            self.bin = pm.Binomial('binomial', n=n*np.ones(k, dtype=np.int),
                                   p=np.ones(k)/n, shape=(1, k), transform=None)
        self._samples = None
        self._cursor = 0

        self.reset()

    def reset(self):
        if self.p is None:
            self.action_values = np.random.uniform(size=self.k)
        else:
            self.action_values = self.p
        self.bin.distribution.p = self.action_values
        if self.t is not None:
            self._samples = self.bin.random(size=self.t).squeeze()
            self._cursor = 0

        self.optimal = np.argmax(self.action_values)

    def pull(self, action):
        return self.sample[action], action == self.optimal

    @property
    def sample(self):
        if self._samples is None:
            return self.bin.random()[0]
        else:
            val = self._samples[self._cursor]
            self._cursor += 1
            return val


class BernoulliBandit(BinomialBandit):
    """
    The Bernoulli distribution models the probability of a single event
    occurring with p probability i.e. get heads on a single p-coin flip. This is
    the special case of the Binomial distribution where N=1.

    In the bandit scenario, this can be used to approximate a hit or miss event,
    such as if a user clicks on a headline, ad, or recommended product.
    """

    def __init__(self, k, p=None, t=None):
        super(BernoulliBandit, self).__init__(k, 1, p=p, t=t)


class PowerBandit():
    def __init__(self, bet):
        self.bet = bet
        self.p_jp = 0.001
        self.p_r7 = self.p_jp + 0.015
        self.p_3b = self.p_r7 + 0.152
        self.p_2b = self.p_3b + 0.522
        self.p_1b = self.p_2b + 2.326
        self.p_ch = self.p_1b + 10.976

    def pull(self):
        self.out_number = np.random.uniform(size=1) * 100
        if self.out_number < self.p_jp:
            return self.bet*1000 - self.bet
        elif self.out_number < self.p_r7 and self.out_number >= self.p_jp:
            return self.bet*100 - self.bet
        elif self.out_number < self.p_3b and self.out_number >= self.p_r7:
            return self.bet*50 - self.bet
        elif self.out_number < self.p_2b and self.out_number >= self.p_3b:
            return self.bet*25 - self.bet
        elif self.out_number < self.p_1b and self.out_number >= self.p_2b:
            return self.bet*10 - self.bet
        elif self.out_number < self.p_ch and self.out_number >= self.p_1b:
            return self.bet*4 - self.bet
        elif self.out_number >= self.p_ch:
            return 0 - self.bet

