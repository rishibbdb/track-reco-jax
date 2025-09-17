#!/usr/bin/env python

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import equinox as eqx

import numpy as np

import sys
sys.path.insert(0, "/home/storage/hans/photondata/gupta/ftpv1/n96_errscale1_32bit_4comp_update/")


import matplotlib.pyplot as plt
from helpers import adjust_plot_1d
from nn_tools import TriplePandelNet

plot_dist = 30
#plot_rho = -3.14
plot_rho = 0.0
#plot_z = -210.0
plot_z = 210.0


key = jax.random.PRNGKey(0)
model = TriplePandelNet(key)
model = jtu.tree_map(lambda x: x.astype(jnp.float32) if eqx.is_array(x) else x, model)
model = eqx.tree_deserialise_leaves("/home/storage/hans/photondata/gupta/ftpv1/n96_errscale1_32bit_4comp_update/cache/new_model_no_penalties_tree_start_epoch_800.eqx", model)
model = jtu.tree_map(lambda x: x.astype(jnp.float64) if eqx.is_array(x) else x, model)

model_eval = eqx.filter_jit(eqx.filter_vmap(model.eval_from_cylindrical_reco_input))

from scipy.special import softmax
cols = plt.cm.plasma(np.linspace(0.1,0.9, 8))

for i in range(4):
    fig, ax = plt.subplots()
    plot_zenith = [0+i*20 for i in range(1, 9)]
    for j,p_zenith in enumerate(plot_zenith):
        xvals = np.linspace(-180, 180, 10000)

        tmp = []
        for azi in xvals:
            tmp.append([plot_dist, plot_rho, plot_z, np.deg2rad(p_zenith), np.deg2rad(azi)])

        xv = jnp.array(tmp, dtype=jnp.float64)
        y = model_eval(xv)
        class_probs, gamma_a, gamma_b = y[:, :4], y[:, 4:8], y[:, 8:]
        class_probs = np.asarray(class_probs)
        gamma_a = np.asarray(gamma_a)
        gamma_b = np.asarray(gamma_b)

        yvals = np.log(gamma_a) / gamma_b
        idx = yvals < 0
        yvals[idx] = 0

        plt.plot(xvals, yvals[:, i], label=f'$\Theta_s = {p_zenith:.0f}$deg', color=cols[j], lw=2)

    outfile = f"./png/anisotropy_azimuth_gamma_{i}_{plot_z:.0f}m_{plot_dist}_m.png"
    plot_args = {'xlim':[-180, 180],
                 'ylim':[1e-3, 1.2 * np.amax(yvals)],
                 'xlabel':'$\Phi_s$ [rad]',
                 'ylabel':'component mode [ns]'}

    adjust_plot_1d(fig, ax, plot_args=plot_args)
    #ax.set_ylim(ymin=1.e-3)
    if i == 2:
        #ax.set_ylim([15, 30])
        #ax.set_ylim([12, 37])
        ax.set_ylim(ymax=200)
    elif i == 0:
        ax.set_ylim([30, 60])

    elif i == 1:
        #ax.set_ylim([10., 30.])
        #ax.set_ylim([20.0, 40.0])
        ax.set_ylim([20.0, 45.0])

    elif i == 3:
        ax.set_ylim(ymax=400)
    #ax.set_yscale('log')
    #ax.axvline(1450+100, color='red', linestyle='dashed')
    #ax.axvline(1450+900, color='red', linestyle='dashed')

    ax.legend(ncol=3, loc='lower left')
#
    plt.title(f"Gamma component {i}, $Z_s$={plot_z:.1f}m, d = {plot_dist:.0f}m, rho={plot_rho}rad")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    del fig


''
