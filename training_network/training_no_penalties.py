import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax
from tqdm import tqdm
from flax import jax_utils
import equinox as eqx

from nn_tools import TriplePandelNet, get_loss_cdf
from tfrecords_utils import parse_numpy, load_table_from_pickle, get_data_iterator_for_jax, tfrecords_reader_dataset

dtype = jnp.float64

display_progress = True

bp = "/home/storage/hans/photondata/gupta/naive/no_penalty_softplus_large/cache/"
model_name = "new_model_no_penalties"

learning_rate = 1.e-4
beta1 = 0.9
beta2 = 0.98

batch_size = 4096
n_epochs = 70
n_steps_per_epoch = 20000

start_epoch = 130
model_serialization_path = os.path.join(bp, f"{model_name}_tree_start_epoch_{start_epoch}.eqx")
optimizer_state_serialization_path = os.path.join(bp, f"{model_name}_optim_state_start_epoch_{start_epoch}.eqx")

# data loading => iterator.
#indir = '/home/storage/hans/photondata/fullazisupport/cartesian/npy_separate_distance/'
#ds_train = parse_numpy(indir=indir, shuffle_buffer=10000*batch_size, batch_size = batch_size, cartesian=True)
#it = get_data_iterator_for_jax(ds_train)

indir = '/home/storage2/hans/photon_data_fine_binned/2024-11-26_tables/tfrecords_shuffled/'
infiles = indir+'*.tfrecords'
ds_train = tfrecords_reader_dataset(infiles, n_readers=32, block_length=32, batch_size=batch_size, shuffle_buffer=batch_size)
it = get_data_iterator_for_jax(ds_train)

# load aux info: table binning.
# open arbitrary pickle to load bin centers
zenith = 99.59406822686046
azimuth = 90.0
infile = f'/home/storage2/hans/photon_data_fine_binned/2024-11-26_tables/pkl/phototable_infinitemuon_zenith{zenith}_azimuth{azimuth}.pkl'
table, bin_info = load_table_from_pickle(infile)
del table
loss_fn = get_loss_cdf(bin_info['dt']['e'], dtype=dtype)

#optim = optax.inject_hyperparams(optax.adamax)(learning_rate=learning_rate, b1=beta1, b2=beta2)
optim = optax.inject_hyperparams(optax.yogi)(learning_rate=learning_rate, b1=beta1, b2=beta2)
#optim = optax.inject_hyperparams(optax.lion)(learning_rate=learning_rate, b1=beta1, b2=beta2)
#schedule = optax.schedules.warmup_constant_schedule(init_value=1.e-5, peak_value=learning_rate, warmup_steps=2000)
#optim = optax.inject_hyperparams(optax.nadam)(learning_rate=schedule, b1=beta1, b2=beta2, nesterov=True)
#optim = optax.inject_hyperparams(optax.yogi)(learning_rate=schedule, b1=beta1, b2=beta2)
#optim = optax.inject_hyperparams(optax.nadam)(learning_rate=learning_rate, b1=beta1, b2=beta2, nesterov=True)

#optim = optax.chain(
#   optax.clip_by_global_norm(0.5),
#   adam,
#)

#fast_solver = optax.inject_hyperparams(optax.adamax)(learning_rate=learning_rate, b1=beta1, b2=beta2)
#optim = optax.lookahead(fast_solver, sync_period, slow_learning_rate)

epoch = start_epoch
total_steps = 0
opt_state = None
if os.path.exists(model_serialization_path):
    print("continue training ...")
    key = jax.random.PRNGKey(0)
    model = TriplePandelNet(key)
    model = eqx.tree_deserialise_leaves(model_serialization_path, model)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    #opt_state = eqx.tree_deserialise_leaves(optimizer_state_serialization_path, opt_state)

else:
    # create new model
    key = jax.random.PRNGKey(0)
    model = TriplePandelNet(key)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))



# things needed for training loop
@eqx.filter_value_and_grad
def compute_loss(model, x, y):
    y_pred = jax.vmap(model.eval_from_training_input)(x)
    return loss_fn(y, y_pred)

@eqx.filter_jit
def make_step(model, x, y, opt_state):
    loss, grads = compute_loss(model, x, y)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state

total_steps = 0
key = jax.random.PRNGKey(0)
for i in range(start_epoch, start_epoch + n_epochs):
    overall_means = []

    opt_state.hyperparams['learning_rate'] = learning_rate
    opt_state.hyperparams['b1'] = beta1
    opt_state.hyperparams['b2'] = beta2

    k = 0
    loss_vals = np.zeros(100)
    batch_it = it
    if display_progress:
        batch_it = tqdm(it, total=n_steps_per_epoch)

    for step, (x, y) in enumerate(batch_it):
        x = jnp.squeeze(x, axis=0)
        y = jnp.squeeze(y, axis=0)

        loss, model, opt_state = make_step(model, x, y, opt_state)

        loss = loss.item()
        loss_vals[k] = loss
        k += 1
        if step % 100 == 0:
            mean_loss = np.mean(loss_vals)
            if step > 1:
                overall_means.append(mean_loss)
                if display_progress:
                    batch_it.set_description(f"step={step}, loss={loss}, mean_loss={mean_loss}")
                    batch_it.update(1) # update progress

            loss_vals = np.zeros(100)
            k = 0

        if step > n_steps_per_epoch:
            break

    total_steps += n_steps_per_epoch
    mean_epoch_loss = np.mean(overall_means)
    print(f"average epoch loss (epoch {i}): {mean_epoch_loss}")

# store progress
next_start_epoch = start_epoch + n_epochs
model_serialization_path = os.path.join(bp, f"{model_name}_tree_start_epoch_{next_start_epoch}.eqx")
optimizer_state_serialization_path = os.path.join(bp, f"{model_name}_optim_state_start_epoch_{next_start_epoch}.eqx")

eqx.tree_serialise_leaves(model_serialization_path, model)
eqx.tree_serialise_leaves(optimizer_state_serialization_path, opt_state)

print("DONE! Jay!")
