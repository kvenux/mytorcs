import numpy as np
import os
import json
import tensorflow as tf
import random
# from vae.vae import ConvVAE, reset_graph
import matplotlib.pyplot as plt
import time
filename = "1107825523.npz"
raw_data = np.load(os.path.join("record", filename))['obs']
print(raw_data.shape)

# def encode_batch(batch_img):
#   simple_obs = np.copy(batch_img).astype(np.float)/255.0
#   simple_obs = simple_obs.reshape(batch_size, 64, 64, 3)
#   mu, logvar = vae.encode_mu_logvar(simple_obs)
#   z = (mu + np.exp(logvar/2.0) * np.random.randn(*logvar.shape))
#   return mu, logvar, z

# def decode_batch(batch_z):
#   # decode the latent vector
#   batch_img = vae.decode(z.reshape(batch_size, z_size)) * 255.
#   batch_img = np.round(batch_img).astype(np.uint8)
#   batch_img = batch_img.reshape(batch_size, 64, 64, 3)
#   return batch_img

# z_size=32
# batch_size=1000 # treat every episode as a batch of 1000!
# learning_rate=0.0001
# kl_tolerance=0.5

# reset_graph()

# vae = ConvVAE(z_size=z_size,
#               batch_size=batch_size,
#               learning_rate=learning_rate,
#               kl_tolerance=kl_tolerance,
#               is_training=False,
#               reuse=False,
#               gpu_mode=True) # use GPU on batchsize of 1000 -> much faster
# model_path_name = "tf_vae"
# vae.load_json(os.path.join(model_path_name, 'vae.json'))
# mu, logvar, z = encode_batch(raw_data)
# gen_data = decode_batch(z)

# fig,ax = plt.subplots(1,1)
# im = ax.imshow(raw_data[6])
fig = plt.figure()
a=fig.add_subplot(1,2,1)
left = plt.imshow(raw_data[10])
a.set_title('Before')
a=fig.add_subplot(1,2,2)
right = plt.imshow(raw_data[10])
a.set_title('After')

cur_index = 10
while True:
    image = raw_data[cur_index]
    left.set_data(raw_data[cur_index])
    right.set_data(raw_data[cur_index])
    fig.canvas.draw_idle()
    plt.pause(0.05)
    cur_index += 1