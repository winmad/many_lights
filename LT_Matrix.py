
# coding: utf-8

# In[13]:

get_ipython().magic(u'matplotlib inline')
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
from scipy import fftpack
from scipy import stats
from scipy import signal

from numba import jit
import time


# In[5]:

n_lights = 512
n_pixels = 512

spp = 135


# Set objects. Light is at (0, 1)-(1, 1). Sensor is at (0, 0)-(1, 0).

# In[6]:

light_sigma = 0.25
d_max = 0.7
pixel_sigma = light_sigma*(1/d_max-1)*0.2

lights = np.exp(-(np.linspace(-0.5, 0.5, 512)/light_sigma)**2/2)
pixels = np.zeros(n_pixels)
objs = [
    np.matrix([[0.2, 0.45], [0.3, 0.4]]).T,
    np.matrix([[0.35, 0.45], [0.4, 0.5]]).T,
    np.matrix([[0.5, 0.3], [0.6, 0.5]]).T,
    np.matrix([[0.9, 0.45], [1.0, 0.4]]).T
]

plt.figure(figsize = (10, 10))
plt.plot([0, 1], [0, 0], linewidth = 20, color = 'b')
plt.plot([0, 1], [1, 1], linewidth = 20, color = 'y')
for obj in objs :
    plt.plot(np.squeeze(np.array(obj[0, :])),             np.squeeze(np.array(obj[1, :])), linewidth = 3 , color='r')


# Compute visibility matrix.

# In[7]:

def intersect(a, b, c, d) :
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    d = np.array(d)
    def ccw(a, b, c) :
        cross = (a[0, :]-b[0, :])*(c[1, :]-b[1, :])
        cross -= (a[1, :]-b[1, :])*(c[0, :]-b[0, :])
        return cross > 0
    c1 = np.logical_xor(ccw(a, b, c), ccw(a, b, d))
    c2 = np.logical_xor(ccw(c, d, a), ccw(c, d, b))
    return np.logical_and(c1, c2)

ax = np.repeat((np.arange(n_pixels)+0.5)/n_pixels, n_lights)
ay = np.zeros(n_pixels*n_lights)
bx = np.tile((np.arange(n_lights)+0.5)/n_lights, n_pixels)
by = np.ones(n_pixels*n_lights)
a = np.vstack((ax, ay))
b = np.vstack((bx, by))
vis = np.zeros(n_pixels*n_lights)
for obj in objs :
    c = np.tile(obj[:, 0], (1, n_pixels*n_lights))
    d = np.tile(obj[:, 1], (1, n_pixels*n_lights))
    vis = np.logical_or(vis, intersect(a, b, c, d))
    
vis = np.logical_not(vis).astype(float)
ltm = vis / np.sqrt((ax-bx)**2+1) * np.tile(lights, n_pixels)
vis = vis.reshape((n_pixels, n_lights))
ltm = ltm.reshape((n_pixels, n_lights))
ref = np.sum(ltm, axis = 1)
wnd = np.ones(n_pixels)
wnd[0:int(n_pixels*0.05)] = 0
wnd[int(n_pixels*0.95):n_pixels] = 0
    
plt.figure()
plt.imshow(ltm, cmap = 'gray');


# # Row Column Sampling Test

# In[8]:

def row_col_sampling(ltm, nr, nc) :
    def dist(a, b) :
        la = np.linalg.norm(a, axis = 0)
        lb = np.linalg.norm(b, axis = 0)
        na = a / la
        nb = b / lb
        m_na = np.repeat(na, len(lb), axis = 1)
        m_nb = np.tile(nb, len(la))
        m_la = np.repeat(la, len(lb), axis = 0)
        m_lb = np.tile(lb, len(la))
        dist_mat = np.sum((m_na-m_nb)**2, axis = 0)*m_la*m_lb
        dist_mat = dist_mat.reshape(len(la), len(lb))
        return dist_mat
    def naive_cluster(rows, nc) :
        center_indices = np.random.permutation(np.arange(0, rows.shape[1]))
        center_indices = center_indices[0:nc]
        centers = rows[:, center_indices]
        dist_mat = dist(centers, rows)
        cluster_indices = np.argmin(dist_mat, axis = 0)
        return center_indices, cluster_indices
    def sampling_cluster(rows, nc) :
        dist_mat = dist(rows, rows)
        probs = np.sum(dist_mat, axis = 0)
        probs /= np.sum(probs)
        rand_gen = stats.rv_discrete(values = (np.arange(len(probs)), probs))
        weights = dict()
        center_indices = np.zeros(nc)
        current_nc = 0
        while current_nc < nc :
            index = rand_gen.rvs()
            if not(index in weights) :
                center_indices[current_nc] = index
                current_nc += 1
                weights[index] = 1 / probs[index]
            else :
                weights[index] += 1 / probs[index]
        centers = rows[:, center_indices.astype(int)]
        for i in range(0, nc) :
            centers[:, i] *= weights[center_indices[i]]
        dist_mat = dist(centers, rows)
        cluster_indices = np.argmin(dist_mat, axis = 0)
        return center_indices, cluster_indices
        
    def cluster(rows, nc) :
        return naive_cluster(rows, nc)
    indices = np.arange(0, ltm.shape[0])
    row_indices = np.zeros(nr)
    patch_size = len(indices) / nr
    for i in range(0, nr) :
        candidates = indices[i*patch_size:np.min(((i+1)*patch_size, len(indices)))]
        row_indices[i] = np.random.choice(candidates)
    row_indices = row_indices.astype(int)
    pixels = np.zeros(ltm.shape[0])
# Sample rows
    rows = ltm[row_indices, :]
    center_indices, cluster_indices = cluster(rows, nc)
    for ci in range(0, nc) :
        cluster_rows = rows[:, np.array(cluster_indices==ci)]
        cluster_ltm = ltm[:, np.array(cluster_indices==ci)]
        energy = np.linalg.norm(cluster_rows, axis = 0)
        total_energy = sum(energy)
        probs = energy / total_energy
        rand_gen = stats.rv_discrete(values = (np.arange(len(probs)), probs))
        index = rand_gen.rvs()
# Sample columns
        col = cluster_ltm[:, index]
        cluster_pixels = col / energy[index] * total_energy
        pixels += cluster_pixels
    return pixels

nr = 48
nc = 96

pixels = row_col_sampling(ltm, nr, nc)
pixels_filtered = ndimage.filters.gaussian_filter(pixels, pixel_sigma*n_pixels)
plt.figure(figsize = (15, 10))
plt.plot(pixels, label = 'rec');
plt.plot(pixels_filtered, label = 'filtered');
plt.plot(ref, label = 'ref');
plt.legend();

plt.figure(figsize = (15, 2))
plt.imshow(np.vstack((pixels, pixels_filtered, ref)), cmap = 'gray', interpolation='nearest', aspect='auto');
plt.text(1, 0, 'rec', fontsize = 15, color = 'r');
plt.text(1, 1, 'filtered', fontsize = 15, color = 'r')
plt.text(1, 2, 'ref', fontsize = 15, color = 'r');

print 'nsamples =', spp * n_pixels
print 'original l2 =', np.sqrt(sum((pixels-ref)**2 * wnd))
print 'filtered l2 =', np.sqrt(sum((pixels_filtered-ref)**2 * wnd))


# # Lightcuts Test
# * segment tree of [0, n_lights = 512]
# * tree node index ~ [1, 1023]
# * only consider geometric error terms

# In[29]:

import Queue

n_tree_nodes = 2 * n_lights
light_reps = np.zeros(n_tree_nodes)
left_bounds = np.zeros(n_tree_nodes)
right_bounds = np.zeros(n_tree_nodes)
cum_lights = np.zeros(n_lights+1)
cum_lights[1:n_lights+1] = np.cumsum(lights)

# return representative light and light interval for each cluster
def build_light_tree(root, l, r):
    left_bounds[root] = l
    right_bounds[root] = r
    if (r - l) * n_lights > 1:
        build_light_tree(root * 2, l, (l + r) * 0.5)
        build_light_tree(root * 2 + 1, (l + r) * 0.5, r)
        light_reps[root] = light_reps[root * 2 + np.random.randint(2)]
    else:
        light_reps[root] = (l + r) * 0.5   

build_light_tree(1, 0, 1)
#print light_reps

def get_cum_intensity(l, r) :
    r_idx = int(r*(n_lights-1)+0.5)
    l_idx = int(l*(n_lights-1)+0.5)
    return cum_lights[r_idx] - cum_lights[l_idx]

def get_light_index(pos):
    return np.int_(pos * n_lights - 0.5)

def get_pixel_index(pos):
    return np.int_(pos * n_pixels - 0.5)

def get_cluster_intensity(node):
    return get_cum_intensity(left_bounds[node], right_bounds[node])

def calc_cluster_contribution(x, node):
    light_pos = light_reps[node]
    pixel_index = get_pixel_index(x)
    return get_cluster_intensity(node) *         vis[pixel_index][get_light_index(light_pos)] /         np.sqrt((x - light_pos) ** 2 + 1)

def geo_error_term(x, node):
    if left_bounds[node] <= x and x <= right_bounds[node]:
        return 1
    elif x < left_bounds[node]:
        return 1 / np.sqrt((left_bounds[node] - x) ** 2 + 1)
    else:
        return 1 / np.sqrt((x - right_bounds[node]) ** 2 + 1)
    
def error_upper_bound(x, node):
    return get_cluster_intensity(node) * geo_error_term(x, node)

n_samples = np.zeros(n_pixels, dtype = int)

def refine_light_cut(pixel_index):
    x = (pixel_index + 0.5) / n_pixels
    ratio = 0.02
    q = Queue.PriorityQueue([])
    MAX_ERR = 550
    q.put((MAX_ERR - error_upper_bound(x, 1), 1))
    total_radiance = calc_cluster_contribution(x, 1)
    global n_samples
    n_samples[pixel_index] += 1
    while True:
        q_item = q.get()
        max_err = MAX_ERR - q_item[0]
        choose_node = q_item[1]
        if max_err > total_radiance * ratio:
            total_radiance -= calc_cluster_contribution(x, choose_node)
            total_radiance += calc_cluster_contribution(x, choose_node * 2)
            total_radiance += calc_cluster_contribution(x, choose_node * 2 + 1)
            q.put((MAX_ERR - error_upper_bound(x, choose_node * 2), choose_node * 2))
            q.put((MAX_ERR - error_upper_bound(x, choose_node * 2 + 1), choose_node * 2 + 1))
            n_samples[pixel_index] += 1
        else:
            break
    return total_radiance

def calc_pixels():
    pixels = np.zeros(n_pixels)
    for i in range(n_pixels):
        pixels[i] += refine_light_cut(i)
    return pixels

start = time.time()
pixels = calc_pixels()
end = time.time()
print 'running time = ', end - start, 's'

pixels_filtered = ndimage.filters.gaussian_filter(pixels, pixel_sigma*n_pixels)

plt.figure(figsize = (15, 10))
plt.plot(pixels, label = 'rec');
plt.plot(pixels_filtered, label = 'filtered');
plt.plot(ref, label = 'ref');
plt.legend();

plt.figure(figsize = (15, 2))
plt.imshow(np.vstack((pixels, pixels_filtered, ref)), cmap = 'gray', interpolation='nearest', aspect='auto');

plt.text(1, 0, 'rec', fontsize = 15, color = 'r');
plt.text(1, 1, 'filtered', fontsize = 15, color = 'r');
plt.text(1, 2, 'ref', fontsize = 15, color = 'r');
        
print 'nsamples =', sum(n_samples)
print 'original l2 =', np.sqrt(sum((pixels-ref)**2 * wnd))
print 'filtered l2 =', np.sqrt(sum((pixels_filtered-ref)**2 * wnd))


# In[17]:

step = 16
indices = np.arange(0, n_pixels, step)
indices = np.append(indices, n_pixels-1)
pixels_interp_func_spline = scipy.interpolate.UnivariateSpline(indices, pixels[indices])
pixels_interp_func_cubic = scipy.interpolate.interp1d(indices, pixels[indices], kind = 'cubic')
pixels_interp_spline = pixels_interp_func_spline(np.arange(n_pixels))
pixels_interp_cubic = pixels_interp_func_cubic(np.arange(n_pixels))

plt.figure(figsize = (15, 10))
plt.plot(indices, pixels[indices], '*', markersize = 10);
plt.plot(pixels_interp_spline, label = 'spline');
plt.plot(pixels_interp_cubic, label = 'cubic');
plt.plot(ref, label = 'ref');
plt.legend();

plt.figure(figsize = (15, 2))
plt.imshow(np.vstack((pixels_interp_spline, pixels_interp_cubic, ref)), cmap = 'gray', interpolation='nearest', aspect='auto');
plt.text(1, 0, 'spline', fontsize = 15, color = 'r');
plt.text(1, 1, 'cubic', fontsize = 15, color = 'r');
plt.text(1, 2, 'ref', fontsize = 15, color = 'r');

print 'nsamples =', sum(n_samples[indices])
print 'spline l2 =', np.sqrt(sum((pixels_interp_spline-ref)**2 * wnd))
print 'cubic l2 =', np.sqrt(sum((pixels_interp_cubic-ref)**2 * wnd))


# # AA Filtering Test

# In[11]:

imp = 0 # 0--uniform 1--importance
strate_num = 34
spp_sn = 4
spp = spp_sn*strate_num

probs = lights / sum(lights)
rand_gen = stats.rv_discrete(values = (np.arange(len(probs)), probs))

for i in range(n_pixels) :
    p = 1 / float(n_pixels)
    sp_array = np.random.randint(n_lights/strate_num, size=spp)
    for k in range(spp_sn) :
        sp_array[k*strate_num:(k+1)*strate_num] += np.arange(strate_num)*(n_lights/strate_num)
    if imp :
        sp_array = rand_gen.rvs(size = spp)
        p = probs[sp_array]
    pixels[i] = sum(ltm[i, sp_array] / p) / spp
    
pixels_filtered = ndimage.filters.gaussian_filter(pixels, pixel_sigma*n_pixels)
plt.figure(figsize = (15, 10))
plt.plot(pixels, label = 'rec');
plt.plot(pixels_filtered, label = 'filtered');
plt.plot(ref, label = 'ref');
plt.legend();

plt.figure(figsize = (15, 2))
plt.imshow(np.vstack((pixels, pixels_filtered, np.sum(ltm, axis = 1))), cmap = 'gray', interpolation='nearest', aspect='auto');
plt.text(1, 0, 'rec', fontsize = 15, color = 'r');
plt.text(1, 1, 'filtered', fontsize = 15, color = 'r');
plt.text(1, 2, 'ref', fontsize = 15, color = 'r');

print 'nsamples =', spp * n_pixels
print 'original l2 =', np.sqrt(sum((pixels-ref)**2 * wnd))
print 'filtered l2 =', np.sqrt(sum((pixels_filtered-ref)**2 * wnd))


# In[ ]:



