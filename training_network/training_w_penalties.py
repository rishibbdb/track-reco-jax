import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

import jax
#jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import numpy as np
import optax
from tqdm import tqdm
from flax import jax_utils
import equinox as eqx

from nn_tools import TriplePandelNet, get_loss_cdf
from tfrecords_utils import parse_numpy, load_table_from_pickle
from tfrecords_utils import get_data_iterator_for_jax, tfrecords_reader_dataset


dtype = jnp.float32
np_dtype = np.float32

display_progress = True

bp = "/home/storage/hans/photondata/gupta/repo/ftpv1_ncomp4/cache/"
model_name = "new_model_no_penalties"

learning_rate = 1.e-3
beta1 = 0.9
beta2 = 0.98
#beta1 = 0.9
#beta2 = 0.999
batch_size = 1024

# We arbitrarily define one epoch as 20000 gradient update steps.
# There are in total of ~90x10^6 data points in the dataset.
# Divide by batch_size to get the number of steps corresponding to one full pass through
# the dataset (i.e. one true epoch).

n_epochs = 5
n_steps_per_epoch = 10000

start_epoch = 0
model_serialization_path = os.path.join(bp, f"{model_name}_tree_start_epoch_{start_epoch}.eqx")
optimizer_state_serialization_path = os.path.join(bp, f"{model_name}_optim_state_start_epoch_{start_epoch}.eqx")

# create penalties matrix
# distance (w1, w2, w3, a1, a2, a3) and (b1, b2, b3)
p_d = jnp.concatenate([jnp.ones((4, 1)) * 1.0, jnp.ones((4, 1)) * 1.0, jnp.ones((4, 1)) * 1.0], axis=0)
# rho
p_rho = jnp.concatenate([jnp.ones((4, 1)) * 1.0, jnp.ones((4, 1)) * 1.0, jnp.ones((4, 1)) * 1.0], axis=0)
# z
p_z = jnp.concatenate([jnp.ones((4, 1)) * 1.0, jnp.ones((4, 1)) * 1.0, jnp.ones((4, 1)) * 1.0], axis=0)
# zenith
p_zen = jnp.concatenate([jnp.ones((4, 1)) * 1.0, jnp.ones((4, 1)) * 1.0, jnp.ones((4, 1)) * 1.0], axis=0)
# azimuth
p_azi = jnp.concatenate([jnp.ones((4, 1)) * 1.0, jnp.ones((4, 1)) * 1.0, jnp.ones((4, 1)) * 1.0], axis=0)
# combine
penalties = jnp.concatenate([p_d, p_rho, p_z, p_zen, p_azi], axis=1) # (12, 5) = (n_outputs, n_inputs)

penalty_scale = 1.e-9

# data loading => iterator.
#indir = '/home/storage/hans/photondata/fullazisupport/cartesian/npy_separate_distance/'
#ds_train = parse_numpy(indir=indir, shuffle_buffer=10000*batch_size, batch_size = batch_size, cartesian=True)
#it = get_data_iterator_for_jax(ds_train)

indir = '/home/storage/hans/datasets/phototable/ftp-v1_flat/02-26-2025/tfrecords_shuffled/'
infiles = indir+'*.tfrecords'
n_readers = 32
shuffle_fac = 100
ds_train = tfrecords_reader_dataset(infiles, n_readers=32, block_length=batch_size * shuffle_fac // n_readers,
        batch_size=batch_size, shuffle_buffer=shuffle_fac * batch_size)
it = get_data_iterator_for_jax(ds_train, np_dtype=np_dtype)

# load aux info: table binning.
# open arbitrary pickle to load bin centers
zenith = 99.59406822686046
azimuth = 90.0
infile = f'/home/storage/hans/datasets/phototable/ftp-v1_flat/02-26-2025/pkl/phototable_infinitemuon_zenith{zenith}_azimuth{azimuth}.pkl'
table, bin_info = load_table_from_pickle(infile)
del table

time_bin_edges = bin_info['dt']['e']
time_bin_edges[0] = 1.e-15

loss_fn = get_loss_cdf(time_bin_edges, dtype=dtype)

optim = optax.inject_hyperparams(optax.yogi)(learning_rate=learning_rate, b1=beta1, b2=beta2)
#optim = optax.inject_hyperparams(optax.adamax)(learning_rate=learning_rate, b1=beta1, b2=beta2)
#optim = optax.inject_hyperparams(optax.lion)(learning_rate=learning_rate, b1=beta1, b2=beta2)

epoch = start_epoch
total_steps = 0
opt_state = None
if os.path.exists(model_serialization_path):
    print("continue training ...")
    key = jax.random.PRNGKey(0)
    model = TriplePandelNet(key)
    # need to cast model and opt state appropriate if loading a float32 model but want to train in float64
    # or vice-versa.
    #model = jtu.tree_map(lambda x: x.astype(jnp.float64) if eqx.is_array(x) else x, model)
    model = eqx.tree_deserialise_leaves(model_serialization_path, model)

    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    #opt_state = jtu.tree_map(lambda x: x.astype(jnp.float64) if eqx.is_array(x) else x, opt_state)
    opt_state = eqx.tree_deserialise_leaves(optimizer_state_serialization_path, opt_state)

    #model = jtu.tree_map(lambda x: x.astype(jnp.float32) if eqx.is_array(x) else x, model)
    #opt_state = jtu.tree_map(lambda x: x.astype(jnp.float32) if eqx.is_array(x) else x, opt_state)

else:
    # create new model
    key = jax.random.PRNGKey(0)
    model = TriplePandelNet(key)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))


# things needed for training loop
def compute_loss(model, x, y, key):
    y_pred = jax.vmap(model.eval_from_training_input)(x)

    # transform x to cylindrical coordinates
    x = jax.vmap(model.cartesian2cylindrical)(x)

    # sample space for smoothness penalty
    key = jax.random.split(key, 7)

    # sample new z values
    z = jax.random.uniform(key[0], minval=-800, maxval=800, shape=x.shape[0]*2)
    z = jnp.expand_dims(z, axis=1)

    # sample new dist values
    n_nearby = 50
    d1 = jax.random.uniform(key[1], minval=1.0, maxval=500, shape=x.shape[0]*2-n_nearby)
    d2 = jax.random.uniform(key[5], minval=1.0, maxval=15, shape=n_nearby)
    d = jnp.concatenate([d1, d2], axis=0)
    d = jnp.expand_dims(d, axis=1)

    # sample rho
    rho = jax.random.uniform(key[2], minval=-jnp.pi, maxval=jnp.pi, shape=x.shape[0]*2)
    rho = jnp.expand_dims(rho, axis=1)

    # sample zenith
    zen = jax.random.uniform(key[3], minval=0.0, maxval=jnp.pi, shape=x.shape[0]*2)
    zen = jnp.expand_dims(zen, axis=1)

    # sample azimuth
    azi = jax.random.uniform(key[4], minval=0.0, maxval=2*jnp.pi, shape=x.shape[0]*2)
    azi = jnp.expand_dims(azi, axis=1)

    x_p = jnp.concatenate([d, rho, z, zen, azi], axis=1)

    # merge with training sample points
    x_p = jnp.concatenate([x_p, x], axis=0)

    model_loss = loss_fn(y, y_pred)

    penalty_loss = jax.vmap(model.smoothness_penalty_wrt_cylindrical_reco_input, (0, None), 0)(x_p, penalties)
    penalty_loss = penalty_scale * jnp.mean(penalty_loss)

    return (model_loss + penalty_loss, (model_loss, penalty_loss))
    #return (model_loss, (model_loss, penalty_loss))

compute_loss_value_and_grad = eqx.filter_value_and_grad(compute_loss, has_aux=True)

@eqx.filter_jit
def make_step(model, x, y, opt_state, key):
    ((loss, loss_info), grads) = compute_loss_value_and_grad(model, x, y, key)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state, loss_info

total_steps = 0
key = jax.random.PRNGKey(0)
average_freq = 100
for i in range(start_epoch, start_epoch + n_epochs):
    overall_means = []
    overall_means_model = []
    overall_means_penalty = []
    opt_state.hyperparams['learning_rate'] = learning_rate
    opt_state.hyperparams['b1'] = beta1
    opt_state.hyperparams['b2'] = beta2

    k = 0
    loss_vals = np.zeros(average_freq)
    loss_vals_model = np.zeros(average_freq)
    loss_vals_penalty = np.zeros(average_freq)

    batch_it = it
    if display_progress:
        batch_it = tqdm(it, total=n_steps_per_epoch)

    for step, (x, y) in enumerate(batch_it):
        x = jnp.squeeze(x, axis=0)
        y = jnp.squeeze(y, axis=0)

        key, subkey = jax.random.split(key)

        loss, model, opt_state, loss_info = make_step(model, x, y, opt_state, subkey)

        # split loss into model and penalty contributions
        model_loss, penalty_loss = loss_info

        loss = loss.item()
        loss_vals[k] = loss
        loss_vals_model[k] = model_loss
        loss_vals_penalty[k] = penalty_loss
        k += 1
        if step % average_freq == 0:
            mean_loss = np.mean(loss_vals)
            mean_loss_model = np.mean(loss_vals_model)
            mean_loss_penalty = np.mean(loss_vals_penalty)
            if step > 1:
                overall_means.append(mean_loss)
                overall_means_model.append(mean_loss_model)
                overall_means_penalty.append(mean_loss_penalty)
                if display_progress:
                    batch_it.set_description(f"step={step}, mean_loss=({mean_loss:.3f}, {mean_loss_model:.3f}, {mean_loss_penalty:.3e})")
                    batch_it.update(1) # update progress

            loss_vals = np.zeros(average_freq)
            loss_vals_model = np.zeros(average_freq)
            loss_vals_penalty = np.zeros(average_freq)
            k = 0

        if step > n_steps_per_epoch:
            break

    total_steps += n_steps_per_epoch
    mean_epoch_loss = np.mean(overall_means)
    mean_epoch_loss_model = np.mean(overall_means_model)
    mean_epoch_loss_penalty = np.mean(overall_means_penalty)
    print(f" >>>> average epoch loss (epoch {i}): ({mean_epoch_loss:.3f}, {mean_epoch_loss_model:.3f}, {mean_epoch_loss_penalty:.3e}) <<<<\n")

# store progress
next_start_epoch = start_epoch + n_epochs
model_serialization_path = os.path.join(bp, f"{model_name}_tree_start_epoch_{next_start_epoch}.eqx")
optimizer_state_serialization_path = os.path.join(bp, f"{model_name}_optim_state_start_epoch_{next_start_epoch}.eqx")

eqx.tree_serialise_leaves(model_serialization_path, model)
eqx.tree_serialise_leaves(optimizer_state_serialization_path, opt_state)

print("DONE! Jay!")
