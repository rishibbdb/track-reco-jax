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
from helpers import adjust_plot_1d, get_bin_idx, load_table_from_pickle
from nn_tools import TriplePandelNet

import matplotlib.pyplot as plt
from scipy.special import softmax
import copy

from collections import defaultdict

def pdf(x, a, b):
    y = a*b*(1. - np.exp(-b*x))**(a-1) * np.exp(-b*x)
    return jnp.where(x > 0, y, 0)

def pdf_mixed(x, a, b, mix_probs):
    x = jnp.expand_dims(x, axis=0)
    a = jnp.expand_dims(a, axis=-1)
    b = jnp.expand_dims(b, axis=-1)
    mix_probs = jnp.expand_dims(mix_probs, axis=-1)
    return jnp.sum(mix_probs * pdf(x, a, b), axis=0)

# make plots
plot_args = defaultdict(list)

rho = 0.0
depths = [500, 210, -210, -500]

# up to 16m
scale1 = 5
scale2 = 10
scale3 = 50
dists = [3.0, 4.0, 5.3, 6.5, 8, 13]
for dist in dists:
    for z in depths:
        plot_args['plot_dist'].append(dist)
        plot_args['plot_z'].append(z)
        plot_args['plot_rho'].append(rho)
        plot_args['scale1'].append(scale1)
        plot_args['scale2'].append(scale2)
        plot_args['scale3'].append(scale3)

# up to 60m
scale1 = 10
scale2 = 20
scale3 = 50
dists = [16.0, 20.0, 25.0, 35.0, 42.0, 50.0]
for dist in dists:
    for z in depths:
        plot_args['plot_dist'].append(dist)
        plot_args['plot_z'].append(z)
        plot_args['plot_rho'].append(rho)
        plot_args['scale1'].append(scale1)
        plot_args['scale2'].append(scale2)
        plot_args['scale3'].append(scale3)

# up to 150m
scale1 = 20
scale2 = 40
scale3 = 50
dists = [60.0, 75.0, 100.0]
for dist in dists:
    for z in depths:
        plot_args['plot_dist'].append(dist)
        plot_args['plot_z'].append(z)
        plot_args['plot_rho'].append(rho)
        plot_args['scale1'].append(scale1)
        plot_args['scale2'].append(scale2)
        plot_args['scale3'].append(scale3)
#scale1, scale2, scale3 = 30, 50, 50
# up to 400m
scale1 = 30
scale2 = 50
scale3 = 50
dists = [150.0, 200.0, 250.0, 300.0, 400.0]
for dist in dists:
    for z in depths:
        plot_args['plot_dist'].append(dist)
        plot_args['plot_z'].append(z)
        plot_args['plot_rho'].append(rho)
        plot_args['scale1'].append(scale1)
        plot_args['scale2'].append(scale2)
        plot_args['scale3'].append(scale3)

#zenith = 168.28414760510762
#azimuth = 135.0

#zenith = 120.00000000000001
#azimuth = 270.0

zenith = 99.59406822686046
azimuth = 0.0

#zenith = 11.715852394892384
#azimuth = 135.0

infile = f'/home/storage/hans/datasets/phototable/ftp-v1_flat/02-26-2025/pkl/phototable_infinitemuon_zenith{zenith}_azimuth{azimuth}.pkl'
table, bin_info = load_table_from_pickle(infile)

key = jax.random.PRNGKey(0)
model = TriplePandelNet(key)
model = jtu.tree_map(lambda x: x.astype(jnp.float32) if eqx.is_array(x) else x, model)
model = eqx.tree_deserialise_leaves("/home/storage/hans/photondata/gupta/ftpv1/n96_errscale1_32bit_4comp_update/cache/new_model_no_penalties_tree_start_epoch_800.eqx", model)
model = jtu.tree_map(lambda x: x.astype(jnp.float64) if eqx.is_array(x) else x, model)

model_eval = eqx.filter_jit(model.eval_from_cylindrical_reco_input)

def make_dt_plot_w_pdf(dist, rho, z, zenith, azimuth, model, table, bin_info, scale=10, logscale=False,  outfile="tmp.png"):

    y = model_eval(jnp.array([dist, rho, z, np.deg2rad(zenith), np.deg2rad(azimuth)], dtype=jnp.float64))
    weights, gamma_a, gamma_b = y[:4], y[4:8], y[8:]

    i = get_bin_idx(dist, bin_info['dist']['e'])
    j = get_bin_idx(rho, bin_info['rho']['e'])
    k = get_bin_idx(z, bin_info['z']['e'])

    prob_vals = copy.copy(table['values'][i, j, k, :])
    prob_err = copy.copy(np.sqrt(table['weights'][i, j, k, :]))

    tot_prob = np.sum(prob_vals)
    if tot_prob == 0:
        return

    prob_vals /= tot_prob
    prob_vals /= bin_info['dt']['w']

    prob_err /= tot_prob
    prob_err /= bin_info['dt']['w']

    fig, ax = plt.subplots()
    ax.hist(bin_info['dt']['c'], bins=bin_info['dt']['e'], weights=prob_vals, histtype='step', lw=1,
            label=f"(d={dist:.1f}m, $\\rho$ ={rho:.2f} rad, z={z:.0f}m)", color='gray')
    ax.errorbar(bin_info['dt']['c'], prob_vals, yerr = prob_err, lw=0, elinewidth=2,color='k', zorder=100)

    xvals = np.linspace(-50, scale*dist, 1000)
    xv = jnp.array(xvals, dtype=jnp.float64)
    w = weights

    yvals = np.asarray(pdf_mixed(xv, gamma_a, gamma_b, w))
    plt.plot(xvals, yvals, 'r-', zorder=10, linewidth=2, label='4-comp EE')

    plot_args = {'xlim':[-2, scale*dist],
                 'ylim':[0.0, 1.2 * np.amax([np.amax(prob_vals), np.amax(yvals)])],
                 'xlabel':'dt [ns]',
                 'ylabel':'pdf'}

    for i,c in zip(range(4), ['purple', 'plum', 'pink', 'cornflowerblue']):
        tw = w[i]
        yvals = np.asarray(tw * pdf(xv, gamma_a[i], gamma_b[i]))
        plt.plot(xvals, yvals, color=c,linestyle='solid', zorder=8, linewidth=1.5,
                label=f'EE {i}')

    adjust_plot_1d(fig, ax, plot_args=plot_args)

    ax.set_title(f"infinite $\mu$ ($\Theta_s$={zenith:.0f}deg, $\Phi_s$={azimuth:.0f}deg)")
    if logscale:
        plt.yscale('log')
        plt.ylim(ymin=1.e-5)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)


def get_nearest_center_value(dist, z, rho):
    i = get_bin_idx(dist, bin_info['dist']['c'])
    j = get_bin_idx(z, bin_info['z']['c'])
    k = get_bin_idx(rho, bin_info['rho']['c'])

    return bin_info['dist']['c'][i], bin_info['z']['c'][j], bin_info['rho']['c'][k]


for i in range(len(plot_args['plot_dist'])):
#for i in range(1):
    plot_dist, plot_rho, plot_z = plot_args['plot_dist'][i], plot_args['plot_rho'][i], plot_args['plot_z'][i]
    scale1, scale2, scale3 = plot_args['scale1'][i], plot_args['scale2'][i], plot_args['scale3'][i]


    dist, z, rho = get_nearest_center_value(plot_dist, plot_z, plot_rho)
    print("generating plot:", dist, z, rho)
    outfile=f'./png_tmp/linear_dist_{dist:04.1f}m_z_{z:.1f}m_rho_{rho:.1f}_scale_{scale1}.png'
    make_dt_plot_w_pdf(dist, rho, z, zenith, azimuth, model, table, bin_info, scale=scale1, outfile=outfile)
    outfile=f'./png_tmp/linear_dist_{dist:04.1f}m_z_{z:.1f}m_rho_{rho:.1f}_scale_{scale2}.png'
    make_dt_plot_w_pdf(dist, rho, z, zenith, azimuth, model, table, bin_info, scale=scale2, outfile=outfile)
    outfile=f'./png_tmp/log_dist_{dist:04.1f}m_z_{z:.1f}m_rho_{rho:.1f}_scale_{scale3}.png'
    make_dt_plot_w_pdf(dist, rho, z, zenith, azimuth, model, table, bin_info, scale=scale3, logscale=True, outfile=outfile)
