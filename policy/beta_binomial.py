import torch
from torch.distributions import constraints
from torch.distributions import Distribution, Beta
from torch.distributions.utils import broadcast_all

__all__ = ['BetaBinomial']


class BetaBinomial(Distribution):
    r"""    
    
    Creates a Beta-Binomial distribution parameterized by :attr:`total_count`,
    :attr:`alpha` and :attr:`beta`. :attr:`total_count`, :attr:`alpha`, and :attr:`beta` must be
    broadcastable with each other.

    .. note:: The beta-binomial distribution is the binomial distribution in which the 
              probability of success at each of n trials is not fixed but randomly drawn from 
              a beta distribution. See https://en.wikipedia.org/wiki/Beta-binomial_distribution#As_a_compound_distribution

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterinistic")
        >>> m = BetaBinomial(100, torch.ones(4), torch.ones(4))
        >>> m.sample()
        tensor([13., 15., 90., 83.])

    Args:
        total_count (int or Tensor): number of Bernoulli trials 
        alpha (Tensor): alpha parameter of the beta distribution 
        beta (Tensor): beta parameter of the beta distribution
    """
    arg_constraints = {
        'total_count': constraints.nonnegative_integer,
        'alpha': constraints.positive,
        'beta': constraints.positive
    }
    has_enumerate_support = True

    def __init__(self, total_count, alpha, beta, validate_args=None):
        self.total_count, self.alpha, self.beta = broadcast_all(total_count, alpha, beta)
        # self.total_count = self.total_count.type_as(self.alpha)

        batch_shape = self.total_count.size()
        self._beta_distribution = Beta(self.alpha, self.beta, validate_args=validate_args)
        super(BetaBinomial, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(BetaBinomial, _instance)
        batch_shape = torch.Size(batch_shape)
        new.total_count = self.total_count.expand(batch_shape)
        new.alpha = self.alpha.expand(batch_shape)
        new.beta = self.beta.expand(batch_shape)
        super(BetaBinomial, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @constraints.dependent_property
    def support(self):
        return constraints.integer_interval(0, self.total_count)

    @property
    def mean(self):
        return self.total_count * self.alpha / (self.alpha + self.beta)

    @property
    def mode(self):
        return super().mode

    @property
    def variance(self):
        n, a, b = self.total_count, self.alpha, self.beta
        return n * a * b * (n + a + b) / (a + b)**2 / (a + b + 1)

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            probs = self._beta_distribution.sample(sample_shape)
            return torch.binomial(self.total_count.expand(shape), probs)

    def log_prob(self, value):

        if self._validate_args:
            self._validate_sample(value)

        def _log_beta(x, y):
            return torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x + y)

        x, n, a, b = value, self.total_count, self.alpha, self.beta

        # logit = log(p)
        #       = log(B(x + a, n - x + b)) - log(B(a, b)) + log(n!) - log(x!) - log((n - x)!)
        #       = log(B(x + a, n - x + b)) - log(B(a, b)) + lgamma(n+1) - lgamma(x+1) - lgamma(n - x + 1)

        lgamma_n = torch.lgamma(n + 1)
        lgamma_x = torch.lgamma(x + 1)
        lgamma_nmx = torch.lgamma(n - x + 1)
        logits = _log_beta(x + a, n - x + b) - _log_beta(a, b) + lgamma_n - lgamma_x - lgamma_nmx
        return logits

    def entropy(self):
        total_count = int(self.total_count.max())
        if not self.total_count.min() == total_count:
            raise NotImplementedError("Inhomogeneous total count not supported by `entropy`.")

        log_prob = self.log_prob(self.enumerate_support(False))
        return -(torch.exp(log_prob) * log_prob).sum(0)

    def enumerate_support(self, expand=True):
        total_count = int(self.total_count.max())
        if not self.total_count.min() == total_count:
            raise NotImplementedError("Inhomogeneous total count not supported by `enumerate_support`.")
        values = torch.arange(1 + total_count, dtype=self.alpha.dtype, device=self.alpha.device)
        values = values.view((-1, ) + (1, ) * len(self._batch_shape))
        if expand:
            values = values.expand((-1, ) + self._batch_shape)
        return values


if __name__ == '__main__':
    total_count = 100
    alpha = torch.randn(2, 48, 48).abs() + 1
    beta = torch.randn(2, 48, 48).abs() + 1

    beta_binomial = BetaBinomial(total_count, alpha, beta)
    sample = beta_binomial.sample((7, ))
    beta_binomial.log_prob(sample)

    BetaBinomial(total_count, 2, 2).log_prob(torch.arange(total_count + 1).type(torch.float32))
