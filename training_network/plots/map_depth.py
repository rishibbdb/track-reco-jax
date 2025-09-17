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

plot_dist = 30.0
#plot_rho = -1.0
plot_rho = 0.0
plot_azimuth = 40.80309989368208

key = jax.random.PRNGKey(0)
model = TriplePandelNet(key)
model = jtu.tree_map(lambda x: x.astype(jnp.float32) if eqx.is_array(x) else x, model)
model = eqx.tree_deserialise_leaves("/home/storage/hans/photondata/gupta/ftpv1/n96_errscale1_32bit_4comp_update/cache/new_model_no_penalties_tree_start_epoch_800.eqx", model)
model = jtu.tree_map(lambda x: x.astype(jnp.float64) if eqx.is_array(x) else x, model)

from scipy.special import softmax
cols = plt.cm.plasma(np.linspace(0.1,0.9, 8))

model_eval = eqx.filter_jit(eqx.filter_vmap(model.eval_from_cylindrical_reco_input))

for i in range(4):
    fig, ax = plt.subplots()
    plot_zenith = np.array([j*20 for j in range(1, 9)]).astype(float)
    for j,p_zenith in enumerate(plot_zenith):
        xvals = np.linspace(1400, 2500, 10000).astype(float)

        tmp = []
        for depth in xvals:
            tmp.append([plot_dist, plot_rho, 1948.07-depth, np.deg2rad(p_zenith), np.deg2rad(plot_azimuth)])

        xv = jnp.array(tmp, dtype=jnp.float64)
        y = model_eval(xv)

        class_probs, gamma_a, gamma_b = y[:, :4], y[:, 4:8], y[:, 8:]
        class_probs = np.asarray(class_probs)
        gamma_a = np.asarray(gamma_a)
        gamma_b = np.asarray(gamma_b)

        yvals = np.log(gamma_a) / gamma_b

        plt.plot(xvals, yvals[:, i], label=f'$\Theta_s = {p_zenith:.0f}$deg', color=cols[j], lw=2)

    outfile = f"./png/depth_plot_azimuth_gamma_{i}_{plot_azimuth:.0f}deg_{plot_rho:.1f}.png"
    plot_args = {'xlim':[1400, 2500],
                 'ylim':[1e-3, 1.2 * np.amax(yvals)],
                 'xlabel':'depth [m]',
                 'ylabel':'component mode [ns]'}

    adjust_plot_1d(fig, ax, plot_args=plot_args)
    if i == 1:
        ax.set_ylim(ymax=100)
    elif i == 0:
        ax.set_ylim(ymax=125)

    elif i == 2:
        ax.set_ylim(ymax=300)

    elif i == 3:
        ax.set_ylim(ymax=500)
    #ax.set_yscale('log')
    #ax.axvline(1450+100, color='red', linestyle='dashed')
    #ax.axvline(1450+900, color='red', linestyle='dashed')

    plt.title(f"Gamma component {i}, $\Phi_s=${plot_azimuth:.1f}deg, d = {plot_dist:.0f}m, rho={plot_rho}rad")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    del fig


'''
for i in range(3):
    fig, ax = plt.subplots()
    plot_zenith = np.array([j*20 for j in range(1, 9)])
    for j,p_zenith in enumerate(plot_zenith):
        xvals = np.linspace(1450, 2450, 10000)
        x = tf.constant([[plot_dist/1000, (1948.07-z)/1000, plot_rho/np.pi, np.cos(np.deg2rad(p_zenith)), np.deg2rad(plot_azimuth)/np.pi] for z in xvals])
        y_pred = model.predict(x)

        logits, gamma_a, gamma_b = convert_pars(y_pred)
        class_probs = softmax(logits, axis=1)
        #yvals = (gamma_a - 1) / gamma_b
        yvals = class_probs

        plt.plot(xvals, yvals[:, i], label=f'$\Theta_s = {p_zenith:.0f}$deg', color=cols[j], lw=2)
    outfile = f"./png/depth_plot_component_weight_azimuth_gamma_{i}_{plot_azimuth:.0f}deg.png"
    plot_args = {'xlim':[1500, 2400],
                 'ylim':[0, 1],
                 'xlabel':'depth [m]',
                 'ylabel':'component weight'}

    adjust_plot_1d(fig, ax, plot_args=plot_args)
    #ax.set_ylim(ymin=1.e-3)
    #if i == 0:
    #    ax.set_ylim(ymax=125)
    #elif i == 1:
    #    ax.set_ylim(ymax=200)
    #ax.set_yscale('log')
    ax.axvline(1450+100, color='red', linestyle='dashed')
    ax.axvline(1450+900, color='red', linestyle='dashed')

    plt.title(f"Gamma component {i}, $\Phi_s=${plot_azimuth:.1f}deg, d = {plot_dist:.0f}m, rho={plot_rho}rad")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    del fig
'''
