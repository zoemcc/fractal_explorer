# Import libraries for simulation
import tensorflow as tf
import numpy as np

# Imports for visualization
import PIL.Image
from cStringIO import StringIO
from IPython.display import clear_output, Image, display
import scipy.ndimage as nd
from skimage import measure
import scipy
from IPython import embed
import matplotlib.pyplot as plt
from numpy.linalg import norm

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
    #print img
    img[a==a.max()] = 0
    a = img
    a = np.uint8(np.clip(a, 0, 255))
    print a.shape
    #f = StringIO()
    #PIL.Image.fromarray(a).save(f, fmt)
    #display(Image(data=f.getvalue()))
    scipy.misc.imsave(filename, a)

def DisplayIntExt(a, filename):

    img_intensity = (((a + 1) / 2) * 255).astype(np.int)
    img = np.concatenate([img_intensity] * 3, 2).astype(np.uint8)
    scipy.misc.imsave(filename, img)

def interior_exterior_map(a):
    img = np.copy(a)
    img[a==a.max()] = -1
    img[a != a.max()] = 1
    return img

def add_pendant(img_int_ext, location, radius, ext_radius):

    new_img_ext = np.copy(img_int_ext)
    for i in xrange(img_int_ext.shape[0]):
        for j in xrange(img_int_ext.shape[1]):
            dist = norm(np.array((i, j)) - location)
            if dist < radius + ext_radius and dist > radius - ext_radius:
                new_img_ext[i, j] = -1
    return new_img_ext

            


def contour_to_int_ext_map(contour, gridX, gridY):




    #img = np.copy(a)
    #img[a==a.max()] = -1
    #img[a != a.max()] = 1
    #return img
    return None


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

    img_int_ext = interior_exterior_map(ns_evaled)
    print "img_int_ext.max, %f, img_int_ext.min: %f" % \
        (img_int_ext.max(), img_int_ext.min())

    location = (X.shape[0] / 2, (4 * X.shape[1]) / 5)
    radius = (X.shape[0] / 10)
    ext_radius = (X.shape[0] / 70)
    img_int_ext_pendant = add_pendant(img_int_ext, location, radius, ext_radius)

    DisplayFractal(img_int_ext_pendant, "mandelbrot_int_ext_pendant.png")

    img_int_ext_pendant_noend = np.copy(img_int_ext_pendant)
    for i in xrange(X.shape[1] / 5):
        for j in xrange(X.shape[0]):
            img_int_ext_pendant_noend[j, i] = 1

    DisplayFractal(img_int_ext_pendant_noend, "mandelbrot_int_ext_pendant_noend.png")

    img_int_ext_pendant_noend_bigmiddle = np.copy(img_int_ext_pendant_noend)
    location_bigmiddle = np.array((X.shape[0] / 2, int(X.shape[1] / 2.5)))
    radius_bigmiddle = X.shape[1] / 25
    for i in xrange(X.shape[1]):
        for j in xrange(X.shape[0]):
            dist = norm(np.array((j, i)) - location_bigmiddle)
            if dist < radius_bigmiddle:
                img_int_ext_pendant_noend_bigmiddle[j, i] = -1

    DisplayFractal(img_int_ext_pendant_noend_bigmiddle, "mandelbrot_int_ext_pendant_noend_bigmiddle.png")

    
    print "img_int_ext: "
    print img_int_ext
    DisplayFractal(img_int_ext, "mandelbrot_int_ext.png")
    contours = measure.find_contours(img_int_ext_pendant_noend_bigmiddle, 0.0)
    len_contours = [len(contour_i) for contour_i in contours]
    print len_contours
    contours_sorted_by_size = sorted(contours, key=len)
    len_contours = [len(contour_i) for contour_i in contours]
    print len_contours
    bigcontour = contours_sorted_by_size[0]
    #embed()

    #bigcontour_int_ext = contour_to_int_ext_map(bigcontour, X, Y)

    #embed()

    fig, ax = plt.subplots()
    #ax.imshow(img_int_ext, interpolation='nearest', cmap=plt.cm.gray)
    #for n, contour in enumerate(contours):
        #ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
    ax.plot(bigcontour[:, 1], bigcontour[:, 0], linewidth=2)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])

    import svgwrite

    dwg = svgwrite.Drawing('mandelbrot.svg', profile='tiny')
    for j in [-1, -2]:
        contour = contours_sorted_by_size[j]
        for i in xrange(contour.shape[0] - 1):
            #print tuple(bigcontour[i])
            dwg.add(dwg.line(tuple(contour[i]), tuple(contour[i + 1]), \
                    stroke=svgwrite.rgb(10, 10, 16, '%')))
    #dwg.add(dwg.text('Test', insert=(0, 0.2), fill='red'))
    dwg.save()
    #plt.show()

    error = 100 * np.abs(mus_evaled - ns_evaled)
    #print error.shape
    error = error.reshape(list(error.shape)+[1])
    #print error.shape
    error_img = np.concatenate([error, error, error], 2)
    error_img = np.uint8(np.clip(error_img, 0, 255))
    scipy.misc.imsave('mandelbrot_errors.png', error_img)


if __name__ == '__main__':
    main()
