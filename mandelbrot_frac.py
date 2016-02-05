# Import libraries for simulation
import tensorflow as tf
import numpy as np

# Imports for visualization
import PIL.Image
from cStringIO import StringIO
from IPython.display import clear_output, Image, display
import scipy.ndimage as nd
import scipy

MAX_ITERS = 200

def DisplayFractal(a, filename):
    """Display an array of iteration counts as a
       colorful picture of a fractal."""
    print type(a)
    a_cyclic_inter = (6.28*a/20.0)
    a_cyclic = a_cyclic_inter.reshape(list(a.shape)+[1])
    img = np.concatenate([10+200*np.cos(a_cyclic),
                          30+500*np.sin(a_cyclic),
                          155-800*np.cos(a_cyclic)], 2)
    print img
    img[a==a.max()] = 0
    a = img
    a = np.uint8(np.clip(a, 0, 255))
    print a.shape
    #f = StringIO()
    #PIL.Image.fromarray(a).save(f, fmt)
    #display(Image(data=f.getvalue()))
    scipy.misc.imsave(filename, a)

def main():

# Use NumPy to create a 2D array of complex numbers on [-2,2]x[-2,2]

    Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]
    print 'Y shape: ', Y.shape
    print 'X shape: ', X.shape
    Z = X+1j*Y

    xs = tf.constant(Z.astype("complex64"))
    zs = tf.Variable(xs)
    ns = tf.Variable(tf.zeros_like(xs, "float32"))
    not_diverged = tf.Variable(np.ones(Z.shape, dtype=np.bool))
    Z_mod_at_div = tf.Variable(2 * tf.ones_like(xs, "float32"))

    for i in range(MAX_ITERS):
        # Compute the new values of z: z^2 + x
        zs_ = zs*zs + xs
        # Have we diverged with this new value?
        cur_mod = tf.complex_abs(zs_)
        not_diverged_ = cur_mod < 4
        # Operation to update the zs and the iteration count.

        # Note: We keep computing zs after they diverge! This
        #       is very wasteful! There are better, if a little
        #       less simple, ways to do this.
        ns_ = ns + tf.cast(not_diverged_, "float32")
        diverged_this_step = tf.logical_and(tf.logical_not(not_diverged_), not_diverged)
        Z_mod_at_div = tf.select(diverged_this_step, cur_mod, Z_mod_at_div)

        zs = zs_

        ns = ns_
        not_diverged = not_diverged_
    mus = tf.select(not_diverged, ns, ns + 1 - tf.log(tf.log(Z_mod_at_div)) / np.log(2))

    with tf.Session() as sess:
        tf.initialize_all_variables().run()


        print 'running!'
        ns_evaled, Z_mod_at_div_evaled, mus_evaled = sess.run([ns, Z_mod_at_div, mus])
        print 'done running!'
        print Z_mod_at_div_evaled
        non_zeros_z_mod = np.where(np.abs(Z_mod_at_div_evaled) > 0.01)
        print non_zeros_z_mod
        print 'max mod: %f, min mod: %f' % (np.max(Z_mod_at_div_evaled), np.min(Z_mod_at_div_evaled[non_zeros_z_mod]))

        print 'diff between mus and ns: '
        diff = mus_evaled - ns_evaled
        print diff
        print 'max: %f, min: %f' % (np.max(diff), np.min(diff))

    DisplayFractal(mus_evaled, 'mandelbrot.png')
    DisplayFractal(ns_evaled, 'mandelbrot_notfrac.png')

    error = 100 * np.abs(mus_evaled - ns_evaled)
    print error.shape
    error = error.reshape(list(error.shape)+[1])
    print error.shape
    error_img = np.concatenate([error, error, error], 2)
    error_img = np.uint8(np.clip(error_img, 0, 255))
    scipy.misc.imsave('mandelbrot_errors.png', error_img)


if __name__ == '__main__':
    main()
