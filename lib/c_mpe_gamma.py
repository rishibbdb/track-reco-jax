import jax.numpy as jnp
import jax
import numpy as np

from jax.scipy.stats.gamma import cdf as gamma_cdf
from jax.scipy.stats.gamma import pdf as gamma_pdf
from jax.scipy.stats.gamma import logpdf as gamma_logpdf
from jax.scipy.stats.norm import pdf as norm_pdf
from jax.scipy.stats.norm import logpdf as norm_logpdf
from jax.scipy.special import logsumexp

from lib.gamma_sf_approx import gamma_sf_fast, c_coeffs, gamma_sf_fast_w_existing_coefficients, log_gamma_sf_fast

#from tensorflow_probability.substrates import jax as tfp
#tfd = tfp.distributions

def c_multi_gamma_mpe_prob_midpoint2(x, mix_probs, a, b, n, sigma=3.0):
    """
    Q < 30
    """
    nmax = 10
    nint1 = 10
    nint2 = 15
    nint3 = 35
    #eps = 1.e-12
    eps = 1.e-6

    x0 = eps
    x_m0 = 0.01
    xvals0 = jnp.linspace(x0, x_m0, 10)

    x_m1 = 0.05
    xvals1 = jnp.linspace(x_m0, x_m1, 10)

    x_m2 = 0.25
    xvals2 = jnp.linspace(x_m1, x_m2, 10)

    x_m25 = 0.75
    xvals25 = jnp.linspace(x_m2, x_m25, 10)

    x_m3 = 2.5
    xvals3 = jnp.linspace(x_m25, x_m3, 10)

    x_m4 = 8.0
    xvals4 = jnp.linspace(x_m3, x_m4, 20)

    #x_m5 = 6000.0
    #xvals5 = jnp.linspace(x_m4, x_m5, 20)

    #xmin = jnp.max(jnp.array([1.5 * eps, x - 4 * sigma]))
    #xmax = jnp.max(jnp.array([xmin+1.5*eps, x + 4 * sigma]))
    #xvals_x = jnp.linspace(xmin, xmax, 30)
    #xvals = jnp.sort(jnp.concatenate([xvals0, xvals1, xvals2, xvals25, xvals3, xvals4, xvals5, xvals_x]))
    xmin = jnp.max(jnp.array([1.5 * eps, x - 10 * sigma]))
    xmax = jnp.max(jnp.array([xmin+1.5*eps, x + 10 * sigma]))
    xvals_x = jnp.linspace(xmin, xmax, 101)
    xvals = jnp.sort(jnp.concatenate([xvals0, xvals1, xvals2, xvals25, xvals3, xvals4, xvals_x]))

    dx = xvals[1:]-xvals[:-1]

    xvals = 0.5*(xvals[:-1]+xvals[1:])
    n_pdf = norm_pdf(xvals, loc=x, scale=sigma)

    a_e = jnp.expand_dims(a, axis=-1)
    b_e = jnp.expand_dims(b, axis=-1)
    mix_probs_e = jnp.expand_dims(mix_probs, axis=-1)

    xvals_e = jnp.expand_dims(xvals, axis=0)
    sfs = jnp.sum(mix_probs_e * gamma_sf_fast(xvals_e, a_e, b_e), axis=0)
    pdfs = jnp.sum(mix_probs_e * jnp.clip(gamma_pdf(xvals_e, a_e, scale=1./b_e), min=0, max=None), axis=0)

    return jnp.sum(n_pdf * n * pdfs * jnp.power(sfs, n-1.0) * dx)

c_multi_gamma_mpe_prob_midpoint2_v = jax.vmap(c_multi_gamma_mpe_prob_midpoint2, (0, 0, 0, 0, 0, None), 0)


#def mpe_pdf_no_conv(x, mix_probs, a, b, n):
#    g_pdf = tfd.MixtureSameFamily(
#                  mixture_distribution=tfd.Categorical(
#                      probs=mix_probs
#                      ),
#                  components_distribution=tfd.Gamma(
#                    concentration=a,
#                    rate=b,
#                    force_probs_to_zero_outside_support=True
#                      )
#    )
#    return n * g_pdf.prob(x) * jnp.power(g_pdf.survival_function(x), n-1.0)
#
#
#def combine(x, mix_probs, a, b, n, sigma):
#    eps = jnp.array(1.e-12)
#    crit = jnp.array(40.0)
#    x_safe = jnp.where(x < eps, eps, x)
#    probs_no_conv = mpe_pdf_no_conv(x_safe, mix_probs, a, b, n)
#    probs_conv = c_multi_gamma_mpe_prob_midpoint2(x, mix_probs, a, b, n, sigma)
#    return jnp.where(x < crit, probs_conv, probs_no_conv)
#
#c_multi_gamma_mpe_prob_combined_v = jax.vmap(combine, (0, 0, 0, 0, 0, None), 0)


def c_multi_gamma_mpe_logprob_midpoint2(x, log_mix_probs, a, b, n, sigma=3.0):
    """
    Q < 30
    """
    nmax = 10
    nint1 = 10
    nint2 = 15
    nint3 = 35
    #eps = 1.e-12
    eps = 1.e-6

    x0 = eps
    x_m0 = 0.01
    xvals0 = jnp.linspace(x0, x_m0, 10)

    x_m1 = 0.05
    xvals1 = jnp.linspace(x_m0, x_m1, 10)

    x_m2 = 0.25
    xvals2 = jnp.linspace(x_m1, x_m2, 10)

    x_m25 = 0.75
    xvals25 = jnp.linspace(x_m2, x_m25, 10)

    x_m3 = 2.5
    xvals3 = jnp.linspace(x_m25, x_m3, 10)

    x_m4 = 8.0
    xvals4 = jnp.linspace(x_m3, x_m4, 20)

    xmin = jnp.max(jnp.array([1.5 * eps, x - 10 * sigma]))
    xmax = jnp.max(jnp.array([xmin+1.5*eps, x + 10 * sigma]))
    xvals_x = jnp.linspace(xmin, xmax, 101)
    xvals = jnp.sort(jnp.concatenate([xvals0, xvals1, xvals2, xvals25, xvals3, xvals4, xvals_x]))

    dx = xvals[1:]-xvals[:-1]

    xvals = 0.5*(xvals[:-1]+xvals[1:])
    log_n_pdf = norm_logpdf(xvals, loc=x, scale=sigma)

    a_e = jnp.expand_dims(a, axis=-1)
    b_e = jnp.expand_dims(b, axis=-1)
    log_mix_probs_e = jnp.expand_dims(log_mix_probs, axis=-1)

    xvals_e = jnp.expand_dims(xvals, axis=0)
    log_pdfs = logsumexp(gamma_logpdf(xvals_e, a_e, scale=1./b_e) + log_mix_probs_e, 0)
    log_sfs = logsumexp(jnp.log(jnp.clip(gamma_sf_fast(xvals_e, a_e, b_e), min=1.e-300)) + log_mix_probs_e, 0)
    #log_sfs = logsumexp(jnp.log(jnp.clip(1.0-gamma_cdf(xvals_e, a_e, b_e), min=1.e-308)) + log_mix_probs_e, 0)

    return logsumexp(log_n_pdf + log_pdfs + (n-1) * log_sfs + jnp.log(dx) + jnp.log(n), 0)

c_multi_gamma_mpe_logprob_midpoint2_v = jax.vmap(c_multi_gamma_mpe_logprob_midpoint2, (0, 0, 0, 0, 0, None), 0)

def c_multi_gamma_mpe_logprob_midpoint2_stable(x, log_mix_probs, a, b, n, sigma=3.0):
    """
    Q < 30
    """
    nmax = 10
    nint1 = 10
    nint2 = 15
    nint3 = 35
    #eps = 1.e-12
    eps = 1.e-6

    x0 = eps
    x_m0 = 0.01
    xvals0 = jnp.linspace(x0, x_m0, 10)[:-1]

    x_m1 = 0.05
    xvals1 = jnp.linspace(x_m0, x_m1, 10)[:-1]

    x_m2 = 0.25
    xvals2 = jnp.linspace(x_m1, x_m2, 10)[:-1]

    x_m25 = 0.75
    xvals25 = jnp.linspace(x_m2, x_m25, 10)[:-1]

    x_m3 = 2.5
    xvals3 = jnp.linspace(x_m25, x_m3, 10)[:-1]

    x_m4 = 8.0
    xvals4 = jnp.linspace(x_m3, x_m4, 20)

    xmin = jnp.max(jnp.array([1.5 * eps, x - 10 * sigma]))
    xmax = jnp.max(jnp.array([xmin+1.5*eps, x + 10 * sigma]))
    xvals_x = jnp.linspace(xmin, xmax, 101)
    xvals = jnp.sort(jnp.concatenate([xvals0, xvals1, xvals2, xvals25, xvals3, xvals4, xvals_x]))

    dx = xvals[1:]-xvals[:-1]

    xvals = 0.5*(xvals[:-1]+xvals[1:])
    log_n_pdf = norm_logpdf(xvals, loc=x, scale=sigma)

    a_e = jnp.expand_dims(a, axis=-1)
    b_e = jnp.expand_dims(b, axis=-1)
    log_mix_probs_e = jnp.expand_dims(log_mix_probs, axis=-1)

    xvals_e = jnp.expand_dims(xvals, axis=0)
    log_pdfs = logsumexp(gamma_logpdf(xvals_e, a_e, scale=1./b_e) + log_mix_probs_e, 0)
    log_sfs = logsumexp(log_gamma_sf_fast(xvals_e, a_e, b_e) + log_mix_probs_e, 0)

    return logsumexp(log_n_pdf + log_pdfs + (n-1) * log_sfs + jnp.log(dx) + jnp.log(n), 0)

c_multi_gamma_mpe_logprob_midpoint2_stable_v = jax.vmap(c_multi_gamma_mpe_logprob_midpoint2_stable, (0, 0, 0, 0, 0, None), 0)
