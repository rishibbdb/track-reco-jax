from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

import equinox as eqx
import jax.numpy as jnp
import jax

from jax import lax
import jax.numpy as jnp
from jax._src.scipy.special import gammaln, xlogy
from jax._src.typing import Array, ArrayLike
from jax.scipy.special import logsumexp

def cdf(x, a, b):
    return jnp.power(1. - jnp.exp(-b*x), a)

def log_cdf(x, a, b):
    return a * log1m_exp(-b*x)

def log_cdf_diff(x1, x2, a, b):
    # x2 > x1: F[x2] - F[x1]

    crit_oob = jnp.log(jnp.finfo(x1.dtype).smallest_normal)+5

    x = a * (log1m_exp(-b*x1) - log1m_exp(-b*x2))
    x_prime = jnp.where(x > -jnp.exp(crit_oob), -jnp.exp(crit_oob), x)
    return log_cdf(x2, a, b) + log1m_exp(x_prime)

def log1m_exp(x):
    """
    Numerically stable calculation
    of the quantity log(1 - exp(x)),
    following the algorithm of
    Machler [1]. This is
    the algorithm used in TensorFlow Probability,
    PyMC, and Stan, but it is not provided
    yet with Numpyro.

    Currently returns NaN for x > 0,
    but may be modified in the future
    to throw a ValueError

    [1] https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    # return 0. rather than -0. if
    # we get a negative exponent that exceeds
    # the floating point representation
    crit = -0.6931472
    arr_x = 1.0 * jnp.array(x)

    crit_oob = jnp.log(jnp.finfo(
        arr_x.dtype).smallest_normal)+5

    oob = arr_x < crit_oob
    mask = arr_x > crit

    more_val = jnp.log(-jnp.expm1(jnp.clip(arr_x, min=crit)))
    less_val = jnp.log1p(-jnp.exp(jnp.clip(arr_x, max=crit)))

    return jnp.where(
        oob,
        -jnp.exp(crit_oob),
        jnp.where(
            mask,
            more_val,
            less_val))

def logpmf(x: ArrayLike, n: ArrayLike, p: ArrayLike) -> Array:
  r"""Multinomial log probability mass function.

  JAX implementation of :obj:`scipy.stats.multinomial` ``logpdf``.

  The multinomial probability distribution is given by

  .. math::

     f(x, n, p) = n! \prod_{i=1}^k \frac{p_i^{x_i}}{x_i!}

  with :math:`n = \sum_i x_i`.

  Args:
    x: arraylike, value at which to evaluate the PMF
    n: arraylike, distribution shape parameter
    p: arraylike, distribution shape parameter

  Returns:
    array of logpmf values.

  See Also:
    :func:`jax.scipy.stats.multinomial.pmf`
  """
  #p, = promote_args_inexact("multinomial.logpmf", p)
  #x, n = promote_args_numeric("multinomial.logpmf", x, n)
  #if not jnp.issubdtype(x.dtype, jnp.integer):
  #  raise ValueError(f"x and n must be of integer type; got x.dtype={x.dtype}, n.dtype={n.dtype}")
  #x = x.astype(p.dtype)
  #n = n.astype(p.dtype)
  logprobs = gammaln(n + 1) + jnp.sum(xlogy(x, p) - gammaln(x + 1), axis=-1)
  return logprobs

def logpmf_logp(x: ArrayLike, n: ArrayLike, logp: ArrayLike) -> Array:
  r"""Multinomial log probability mass function.

  JAX implementation of :obj:`scipy.stats.multinomial` ``logpdf``.

  The multinomial probability distribution is given by

  .. math::

     f(x, n, p) = n! \prod_{i=1}^k \frac{p_i^{x_i}}{x_i!}

  with :math:`n = \sum_i x_i`.

  Args:
    x: arraylike, value at which to evaluate the PMF
    n: arraylike, distribution shape parameter
    p: arraylike, distribution shape parameter

  Returns:
    array of logpmf values.

  See Also:
    :func:`jax.scipy.stats.multinomial.pmf`
  """
  logprobs = gammaln(n + 1) + jnp.sum(x*logp- gammaln(x + 1), axis=-1)
  return logprobs

def get_loss_cdf(time_bin_edges, dtype=jnp.float32):
    time_bin_edges = jnp.array(time_bin_edges.reshape(1, 1, len(time_bin_edges)), dtype=dtype)

    def loss(y_true, y_pred):
        eps = jnp.array(1.e-20)

        a = y_pred[:, 4:8]
        b = y_pred[:, 8:12]

        a_e = jnp.expand_dims(a, axis=-1)
        b_e = jnp.expand_dims(b, axis=-1)

        log_mix_probs = jax.nn.log_softmax(y_pred[:, :4], axis=1)
        log_mix_probs = jnp.expand_dims(log_mix_probs, axis=-1)

        log_probs = log_cdf_diff(time_bin_edges[:, :, :-1], time_bin_edges[:, :, 1:], a_e, b_e)
        log_probs = jnp.clip(log_probs, max=0.0)
        log_probs = logsumexp(log_probs + log_mix_probs, axis=1)

        #probs = cdf(time_bin_edges, a_e, b_e)
        #probs = jnp.sum(mix_probs * probs, axis=1)

        #probs = gm.cdf(time_bin_edges) # nbins x nsamples
        #probs = probs.T # nsamples x nbins

        #probs = probs[:, 1:] - probs[:, :-1]
        #probs = jnp.clip(probs, min=0.0, max=1.0)

        # add some floor for safety.
        #probs = probs + eps * jnp.ones_like(probs)
        #probs = jnp.clip(probs, min=0.0, max=1.0)

        z = logpmf_logp(y_true, jnp.sum(y_true, axis=1), log_probs)
        return -jnp.mean(z)

    return loss

# define model
class TriplePandelNet(eqx.Module):
    layer0: eqx.Module
    layer1: eqx.Module
    layer2: eqx.Module
    layer3: eqx.Module
    layer4: eqx.Module
    layer5: eqx.Module
    layer6: eqx.Module

    def __init__(self, key, hidden_size=96):
        key = jax.random.split(key, 7)
        self.layer0 = eqx.nn.Linear(7, hidden_size, key=key[0, :])
        self.layer1 = eqx.nn.Linear(hidden_size, hidden_size, key=key[1, :])
        self.layer2 = eqx.nn.Linear(hidden_size, hidden_size, key=key[2, :])
        self.layer3 = eqx.nn.Linear(hidden_size, hidden_size, key=key[3, :])
        self.layer4 = eqx.nn.Linear(hidden_size, hidden_size, key=key[4, :])
        self.layer5 = eqx.nn.Linear(hidden_size, hidden_size, key=key[5, :])
        self.layer6 = eqx.nn.Linear(hidden_size+1, 12, key=key[6, :])

    def eval(self, x):
        # x = [dist, cos(rho), sin(rho), z, cos(zenith),
        # sin(zenith)*cos(azimuth), sin(zenith)*sin(azimuth)]
        # dist, z in units of km

        d = x[0:1]

        x = jnp.tanh(self.layer0(x))
        x = jnp.tanh(self.layer1(x)) + x
        x = jnp.tanh(self.layer2(x)) + x
        x = jnp.tanh(self.layer3(x)) + x
        x = jnp.tanh(self.layer4(x)) + x
        x = jnp.tanh(self.layer5(x)) + x

        # outputs
        y = self.layer6(jnp.concatenate([x, d], axis=-1))

        # try to put an ordering on the components
        # by shifting initialization
        to_be_b = y[8:] + jnp.array([-7.62061505, -6.69140313, -5.70345812, -3.89233063])

        x = jnp.concatenate([y[:8], to_be_b], axis=-1)
        return x

    def __call__(self, x):
        return self.eval(x)

    def transform_output_logits(self, y):
        eps = jnp.array(1.e-20)
        # constrain gamma_a
        #a = 1.0 + 20*jax.nn.sigmoid(y[3:6]) + eps
        a = 1.0 + jnp.exp(y[4:8]) + eps
        # constrain gamma_b
        #b = 5.0 * jax.nn.sigmoid(y[6:9])
        b = 1.0 / (1.e4*jax.nn.sigmoid(y[8:12]) + 0.1)
        return jnp.concatenate([y[0:4], a, b])


    def transform_output_probs(self, y):
        eps = jnp.array(1.e-20)
        # get a and b
        y2 = self.transform_output_logits(y)
        # compute weights from logits
        weights = jax.nn.softmax(y[:4])
        # combine
        return jnp.concatenate([weights, y2[4:]])


    def transform_input(self, x):
        # x = [dist, rho, z, zenith, azimuth]
        # distance/z units: m
        # angle units: rad

        km_scale = 1000.
        d = x[0] / km_scale
        rho_x = jnp.cos(x[1])
        rho_y = jnp.sin(x[1])
        z = x[2] / km_scale

        sin_zen = jnp.sin(x[3])
        dir_z = jnp.cos(x[3])

        dir_x = sin_zen * jnp.cos(x[4])
        dir_y = sin_zen * jnp.sin(x[4])

        return jnp.array([d, rho_x, rho_y, z, dir_z, dir_x, dir_y])

    def eval_from_training_input(self, x):
        # x = [dist, rho_x, rho_y, z, dir_z, dir_x, dir_y]
        # distance in units of km
        # rho, dir as cartesian unit vector components

        y = self.eval(x)

        # during training we want predictions to be on logit scale
        return self.transform_output_logits(y)

    def eval_from_cylindrical_reco_input(self, x):
        # x = [dist, rho, z, zenith, azimuth]
        # distance/z units: m
        # angle units: rad

        x = self.transform_input(x)
        y = self.eval(x)

        # this function is used for penalities.
        # we want the second derivatives to be penalized
        # on the probability scale (not logit).
        return self.transform_output_probs(y)

    def cartesian2cylindrical(self, x):
         # x = [dist, rho_x, rho_y, z, dir_z, dir_x, dir_y]
         rho = jnp.atan2(x[2], x[1])
         zenith = jnp.arccos(x[4])
         azimuth = jnp.atan2(x[6], x[5])
         return jnp.array([x[0], rho, x[3], zenith, azimuth])

    def smoothness_penalty_wrt_cylindrical_reco_input(self, x, penalties):
        # penalties : matrix of shape = (9, 5) =  (network_outputs, network_inputs)
        hessian_diag = jnp.diagonal(
                                        jax.jacfwd(jax.jacfwd(self.eval_from_cylindrical_reco_input))(x),
                                        axis1=1, axis2=2
                                    )
        return jnp.sum(penalties * hessian_diag * hessian_diag)
