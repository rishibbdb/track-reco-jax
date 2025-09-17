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
sys.path.insert(0, "/home/storage/hans/photondata/gupta/ftpv1/n96_errscale1_32bit_4comp/")


import matplotlib.pyplot as plt
from helpers import adjust_plot_1d
from nn_tools import TriplePandelNet

pos_z = -210.0
rho = -3.0

#plot_zenith = 95.0
#plot_azimuth = 40.80309989368208

#plot_zenith = 95.0
#plot_azimuth = 139.19690010631788

#plot_zenith = 168.28414760510762
#plot_azimuth = 135.0

plot_zenith = 120.00000000000001
plot_azimuth = 270.0

#plot_zenith = 159.9049673536629
#plot_azimuth = 120.92089564577029

#plot_zenith = 20.095032646337103
#plot_azimuth = 59.0791043542297


key = jax.random.PRNGKey(0)
model = TriplePandelNet(key)
model = jtu.tree_map(lambda x: x.astype(jnp.float32) if eqx.is_array(x) else x, model)
model = eqx.tree_deserialise_leaves("/home/storage/hans/photondata/gupta/ftpv1/n96_errscale1_32bit_4comp/cache/new_model_no_penalties_tree_start_epoch_200.eqx", model)
model = jtu.tree_map(lambda x: x.astype(jnp.float64) if eqx.is_array(x) else x, model)
model_eval = eqx.filter_jit(eqx.filter_vmap(model.eval_from_cylindrical_reco_input))

from scipy.special import softmax

xvals = np.linspace(1, 400, 10000)

pars = []
for d in xvals:
	#x = transform_dimensions(d, rho, pos_z)
	#y = dir_to_xyz(plot_zenith, plot_azimuth)
	pars.append([d, rho, pos_z, np.deg2rad(plot_zenith), np.deg2rad(plot_azimuth)])

x = jnp.array(pars, dtype=jnp.float64)
y = model_eval(x)
class_probs, gamma_a, gamma_b = y[:, :4], y[:, 4:8], y[:, 8:]
class_probs = np.asarray(class_probs)
gamma_a = np.asarray(gamma_a)
gamma_b = np.asarray(gamma_b)

outfile = f"./png/fit_pars_as_fn_of_distance_component_weight_zenith_{plot_zenith:.1f}_azimuth_{plot_azimuth:.1f}.png"
fig, ax = plt.subplots()
for i,c in zip(range(4), ['purple', 'plum', 'pink', 'cornflowerblue']):
    plt.plot(xvals, class_probs[:, i], label=f'Gamma {i}', color=c, lw=2)

plot_args = {'xlim':[-2, 400],
                 'ylim':[0.0, 1.2 * np.amax(class_probs)],
                 'xlabel':'distance [m]',
                 'ylabel':'component weight'}

adjust_plot_1d(fig, ax, plot_args=plot_args)
ax.axvline(x=10, color='red', linestyle='dashed')
plt.title(f"$\Theta_s=${plot_zenith:.1f}deg, $\Phi_s=${plot_azimuth:.1f}deg, z = {pos_z:.0f}m, rho={rho}rad")
plt.tight_layout()
plt.savefig(outfile, dpi=300)
plt.close()
del fig


outfile = f"./png/fit_pars_as_fn_of_distance_component_mode_zenith_{plot_zenith:.1f}_azimuth_{plot_azimuth:.1f}.png"
fig, ax = plt.subplots()
yvals = jnp.log(gamma_a) / gamma_b
for i,c in zip(range(4), ['purple', 'plum', 'pink', 'cornflowerblue']):
    plt.plot(xvals, yvals[:, i], label=f'Gamma {i}', color=c, lw=2)

plot_args = {'xlim':[-2, 400],
                 'ylim':[0.0, 1.2 * np.amax(yvals)],
                 'xlabel':'distance [m]',
                 'ylabel':'component mode [ns]'}

adjust_plot_1d(fig, ax, plot_args=plot_args)
plt.title(f"$\Theta_s=${plot_zenith:.1f}deg, $\Phi_s=${plot_azimuth:.1f}deg, z = {pos_z:.0f}m, rho={rho:.2f}rad")
ax.axvline(x=10, color='red', linestyle='dashed')
plt.tight_layout()
plt.savefig(outfile, dpi=300)
plt.close()
del fig


outfile = f"./png/fit_pars_as_fn_of_distance_component_mode_zoom_zenith_{plot_zenith:.1f}_azimuth_{plot_azimuth:.1f}.png"
fig, ax = plt.subplots()
yvals = jnp.log(gamma_a) / gamma_b
for i,c in zip(range(4), ['purple', 'plum', 'pink', 'cornflowerblue']):
    plt.plot(xvals, yvals[:, i], label=f'Gamma {i}', color=c, lw=2)

plot_args = {'xlim':[5, 50],
                 'ylim':[0.0, 300],
                 'xlabel':'distance [m]',
                 'ylabel':'component mode [ns]'}

adjust_plot_1d(fig, ax, plot_args=plot_args)
plt.title(f"$\Theta_s=${plot_zenith:.1f}deg, $\Phi_s=${plot_azimuth:.1f}deg, z = {pos_z:.0f}m, rho={rho:.2f}rad")
ax.axvline(x=10, color='red', linestyle='dashed')
plt.tight_layout()
plt.savefig(outfile, dpi=300)
plt.close()
del fig

outfile = f"./png/fit_pars_as_fn_of_distance_gamma_a_zenith_{plot_zenith:.1f}_azimuth_{plot_azimuth:.1f}.png"
fig, ax = plt.subplots()
yvals = gamma_a
for i,c in zip(range(4), ['purple', 'plum', 'pink', 'cornflowerblue']):
    plt.plot(xvals, yvals[:, i], label=f'Gamma {i}', color=c, lw=2)

plot_args = {'xlim':[-2, 400],
                 'ylim':[0.0, 1.2 * np.amax(yvals)],
                 'xlabel':'distance [m]',
                 'ylabel':'a (shape)'}

adjust_plot_1d(fig, ax, plot_args=plot_args)
ax.axvline(x=10, color='red', linestyle='dashed')
plt.title(f"$\Theta_s=${plot_zenith:.1f}deg, $\Phi_s=${plot_azimuth:.1f}deg, z = {pos_z:0f}m, rho={rho:.2f}rad")
plt.tight_layout()
plt.savefig(outfile, dpi=300)
plt.close()
del fig

outfile = f"./png/fit_pars_as_fn_of_distance_gamma_a_zoom_zenith_{plot_zenith:.1f}_azimuth_{plot_azimuth:.1f}.png"
fig, ax = plt.subplots()
yvals = gamma_a
for i,c in zip(range(4), ['purple', 'plum', 'pink', 'cornflowerblue']):
    plt.plot(xvals, yvals[:, i], label=f'Gamma {i}', color=c, lw=2)

plot_args = {'xlim':[-2, 50],
                 'ylim':[0.0, 30],
                 'xlabel':'distance [m]',
                 'ylabel':'a (shape)'}

adjust_plot_1d(fig, ax, plot_args=plot_args)
ax.axvline(x=10, color='red', linestyle='dashed')
plt.title(f"$\Theta_s=${plot_zenith:.1f}deg, $\Phi_s=${plot_azimuth:.1f}deg, z = {pos_z:0f}m, rho={rho:.2f}rad")
plt.tight_layout()
plt.savefig(outfile, dpi=300)
plt.close()
del fig


outfile = f"./png/fit_pars_as_fn_of_distance_gamma_b_zenith_{plot_zenith:.1f}_azimuth_{plot_azimuth:.1f}.png"
fig, ax = plt.subplots()
yvals = gamma_b
for i,c in zip(range(4), ['purple', 'plum', 'pink', 'cornflowerblue']):
    plt.plot(xvals, yvals[:, i], label=f'Gamma {i}', color=c, lw=2)

plot_args = {'xlim':[-2, 400],
                 'ylim':[1e-3, 1.2 * np.amax(yvals)],
                 'xlabel':'distance [m]',
                 'ylabel':'b (rate)'}

adjust_plot_1d(fig, ax, plot_args=plot_args)
ax.set_ylim(ymin=1.e-3)
ax.set_yscale('log')
ax.axvline(x=10, color='red', linestyle='dashed')
plt.title(f"$\Theta_s=${plot_zenith:.1f}deg, $\Phi_s=${plot_azimuth:.1f}deg, z = {pos_z:0f}m, rho={rho:.2f}rad")
plt.tight_layout()
plt.savefig(outfile, dpi=300)
plt.close()
del fig


outfile = f"./png/fit_pars_as_fn_of_distance_gamma_b_zoom_zenith_{plot_zenith:.1f}_azimuth_{plot_azimuth:.1f}.png"
fig, ax = plt.subplots()
yvals = gamma_b
for i,c in zip(range(4), ['purple', 'plum', 'pink', 'cornflowerblue']):
    plt.plot(xvals, yvals[:, i], label=f'Gamma {i}', color=c, lw=2)

plot_args = {'xlim':[-2, 50],
                 'ylim':[1e-3, 1.2 * np.amax(yvals)],
                 'xlabel':'distance [m]',
                 'ylabel':'b (rate)'}

adjust_plot_1d(fig, ax, plot_args=plot_args)
ax.set_ylim(ymin=1.e-3)
ax.set_yscale('log')
ax.axvline(x=10, color='red', linestyle='dashed')
plt.title(f"$\Theta_s=${plot_zenith:.1f}deg, $\Phi_s=${plot_azimuth:.1f}deg, z = {pos_z:0f}m, rho={rho:.2f}rad")
plt.tight_layout()
plt.savefig(outfile, dpi=300)
plt.close()
del fig



