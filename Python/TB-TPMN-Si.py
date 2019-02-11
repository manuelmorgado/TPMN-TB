import numpy as np
from matplotlib import pyplot as plt
# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['text.latex.unicode'] = True

#https://gist.github.com/wmedlar/3987bfd8a0f38f7dff4e

def phase(k, neighbors):
    '''
    Determines phase factors of overlap parameters using the assumption that the
    orbitals of each crystal overlap only with those of its nearest neighbor.
    
    args:
        k: a numpy array of shape (3,) that represents the k-point at which to
           calculate phase factors.
        neighbors: a numpy array of shape (4, 3) that represents the four nearest
                   neighbors in the lattice of an atom centered at (0, 0, 0).
                   
    returns:
        A numpy array of shape (4,) containing the (complex) phase factors.
    '''
    
    a, b, c, d = [np.exp(1j * k @ neighbor) for neighbor in neighbors]
    factors = np.array([
        a + b + c + d,
        a + b - c - d,
        a - b + c - d,
        a - b - c + d
    ])
    return (1 / 4) * factors

def band_energies(g, es, ep, vss, vsp, vxx, vxy):
    '''
    Calculates the band energies (eigenvalues) of a material using the
    tight-binding approximation for single nearest-neighbor interactions.

    args:
        g: a numpy array of shape (4,) representing the phase factors with respect
           to a wavevector k and the crystal's nearest neighbors.
        es, ep, vss, vsp, vxx, vxy: empirical parameters for orbital overlap
                                    interactions between nearest neighbors.
                                    
    returns:
        A numpy array of shape (8,) containing the eigenvalues of the
        corresponding Hamiltonian.
    '''

    gc = np.conjugate(g)

    hamiltonian = np.array([
        [         es,  vss * g[0],            0,            0,            0, vsp * g[1], vsp * g[2], vsp * g[3]],
        [vss * gc[0],          es, -vsp * gc[1], -vsp * gc[2], -vsp * gc[3],          0,          0,          0],
        [          0, -vsp * g[1],           ep,            0,            0, vxx * g[0], vxy * g[3], vxy * g[1]],
        [          0, -vsp * g[2],            0,           ep,            0, vxy * g[3], vxx * g[0], vxy * g[1]],
        [          0, -vsp * g[3],            0,            0,           ep, vxy * g[1], vxy * g[2], vxx * g[0]],
        [vsp * gc[1],           0,  vxx * gc[0],  vxy * gc[3],  vxy * gc[1],         ep,         0,           0],
        [vsp * gc[2],           0,  vxy * gc[3],  vxx * gc[0],  vxy * gc[2],          0,        ep,           0],
        [vsp * gc[3],           0,  vxy * gc[1],  vxy * gc[1],  vxx * gc[0],          0,         0,          ep]
    ])

    eigvals = np.linalg.eigvalsh(hamiltonian)
    eigvals.sort()
    return eigvals

def band_structure(params, neighbors, path):
    
    bands = []
    
    for k in np.vstack(path):
        g = phase(k, neighbors)
        # print(g)
        eigvals = band_energies(g, *params)
        bands.append(eigvals)
        
    return np.stack(bands, axis=-1)

def linpath(a, b, n=50, endpoint=True):
    '''
    Creates an array of n equally spaced points along the path a -> b, not inclusive.

    args:
        a: an iterable of numbers that represents the starting position.
        b: an iterable of numbers that represents the ending position.
        n: the integer number of sample points to calculate. Defaults to 50.
        
    returns:
        A numpy array of shape (n, k) where k is the shortest length of either
        iterable -- a or b.
    '''
    # list of n linear spacings between the start and end of each corresponding point
    spacings = [np.linspace(start, end, num=n, endpoint=endpoint) for start, end in zip(a, b)]
    
    # stacks along their last axis, transforming a list of spacings into an array of points of len n
    return np.stack(spacings, axis=-1)

##############
##############

# Es, Ep, Vss, Vsp, Vxx, Vxy (from Papaconstantopoulos' book)
# Ep - Es = 7.20

params = (-5.19278, 1.05825, -2.36233, 1.86401, 2.85882, -0.94687)

# k-points per path
n = 1000

# lattice constant
a = 1

# nearest neighbors to atom at (0, 0, 0)
neighbors = a / 4 *  np.array([
    [1, 1, 1],
    [1, -1, -1],
    [-1, 1, -1],
    [-1, -1, 1]])

# symmetry points in the Brillouin zone
G = 2 * np.pi / a * np.array([0, 0, 0])
L = 2 * np.pi / a * np.array([1/2, 1/2, 1/2])
K = 2 * np.pi / a * np.array([3/4, 3/4, 0])
X = 2 * np.pi / a * np.array([0, 0, 1])
W = 2 * np.pi / a * np.array([1, 1/2, 0])
U = 2 * np.pi / a * np.array([1/4, 1/4, 1])

# k-paths
lambd = linpath(L, G, n, endpoint=False)
delta = linpath(G, X, n, endpoint=False)
x_uk = linpath(X, U, n // 4, endpoint=False)
sigma = linpath(K, G, n, endpoint=True)

bands = band_structure(params, neighbors, path=[lambd, delta, x_uk, sigma])


#matplotlib inline

plt.figure(figsize=(11, 7))

ax = plt.subplot(111)

# remove plot borders
# ax.spines['top'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)

# limit plot area to data
plt.xlim(0, len(bands))
plt.ylim(min(bands[0]) - 1, max(bands[7]) + 1)

# custom tick names for k-points
xticks = n * np.array([0, 0.5, 1, 1.5, 2, 2.25, 2.75, 3.25])
plt.xticks(xticks, ('$L$', '$\Lambda$', '$\Gamma$', '$\Delta$', '$X$', '$U,K$', '$\Sigma$', '$\Gamma$'), fontsize=16)
plt.yticks(fontsize=18)

# horizontal guide lines every 2.5 eV
for y in np.arange(-25, 25, 2.5):
    plt.axhline(y, ls='--', lw=0.3, color='black', alpha=0.3)

# hide ticks, unnecessary with gridlines
plt.tick_params(axis='both', which='both',
                top='off', bottom='off', left='off', right='off',
                labelbottom='on', labelleft='on', pad=5)

plt.xlabel('k-path', fontsize=15)
plt.ylabel('Energy (eV)', fontsize=15)
plt.grid()
plt.text(1350, -18, 'Band structure of Silicon', fontsize=11)

# tableau 10 in fractional (r, g, b)
colors = 1 / 255 * np.array([
    [31, 119, 180],
    [255, 127, 14],
    [44, 160, 44],
    [214, 39, 40],
    [148, 103, 189],
    [140, 86, 75],
    [227, 119, 194],
    [127, 127, 127],
    [188, 189, 34],
    [23, 190, 207]])

for band, color in zip(bands, colors):
    plt.plot(band, lw=2.0, color=color)
plt.savefig('Si bands.png')
plt.show()