import numpy as np
from matplotlib import pyplot as plt

def phase(k, neighbors):
    '''
    Determines phase factors of overlap parameters using the assumption that the
    orbitals of each crystal overlap only with those of its nearest neighbor.
    
    args:
        k: a numpy array of shape (3,) that represents the k-point at which to
           calculate phase factors.
        neighbors: a numpy array of shape (12, 3) that represents the four nearest
                   neighbors in the lattice of an atom centered at (0, 0, 0).
                   
    returns:
        A numpy array of shape (12,) containing the (np.complex) phase factors.
    '''
    
    a, b, c, d, e, f, g, h, i, j, k, l = [np.exp(1j * k @ neighbor) for neighbor in neighbors]
    factors = np.array([
        a + b + c + d + e + f + g + h + i + j + k + l,
        a + b - c - d - e - f - g - h - i - j - k - l,
        a - b + c - d - e - f - g - h - i - j - k - l,
        a - b - c + d - e - f - g - h - i - j - k - l,
        a - b - c - d + e - f - g - h - i - j - k - l,
        a - b - c - d - e + f - g - h - i - j - k - l,
        a - b - c - d - e - f + g - h - i - j - k - l,
        a - b - c - d - e - f - g + h - i - j - k - l,
        a - b - c - d - e - f - g - h + i - j - k - l,
        a - b - c - d - e - f - g - h - i + j - k - l,
        a - b - c - d - e - f - g - h - i - j + k - l,
        a - b - c - d - e - f - g - h - i - j - k + l,

    ])
    return (1 / 12) * factors

def band_energies(sssigma , spsigma , sdsigma , ppsigma , pppi , pdsigma , ddsigma , ddpi , dddelta, pdpi):
    '''
    Calculates the band energies (eigenvalues) of a material using the
    tight-binding approximation for single nearest-neighbor interactions.

    args:
        g: a numpy array of shape (12,) representing the phase factors with respect
           to a wavevector k and the crystal's nearest neighbors.
        sssigma , spsigma , sdsigma , ppsigma , pppi , pdsigma , ddsigma , 
        ddpi , dddelta, pdpi: 
                            empirical parameters for orbital overlap
                            interactions between nearest neighbors.
                                    
    returns:
        A numpy array of shape (8,) containing the eigenvalues of the
        corresponding Hamiltonian.
    '''

    # gc = np.conjugate(g)


    hamiltonian = np.array( [np.array( [sssigma,0,( -2 * ( np.e )**( np.complex( 0,-1 ) * theta ) * spsigma + -2 * ( np.e )**( np.complex( 0,1 ) * theta ) * spsigma ),0,2 * ( 6 )**( 1/2 ) * ( np.e )**( ( np.complex( 0,1 ) * theta + np.complex( 0,2 ) * Phi ) ) * sdsigma,0,( 6 * ( np.e )**( np.complex( 0,-1 ) * theta ) * sdsigma + 6 * ( np.e )**( np.complex( 0,1 ) * theta ) * sdsigma ),0,2 * ( 6 )**( 1/2 ) * ( np.e )**( ( np.complex( 0,-1 ) * theta + np.complex( 0,-2 ) * Phi ) ) * sdsigma,] ),np.array( [0,8 * pppi,0,8 * ( np.e )**( np.complex( 0,-2 ) * Phi ) * pppi,0,0,0,0,0,] ),np.array( [( 2 * ( np.e )**( np.complex( 0,-1 ) * theta ) + 2 * ( np.e )**( np.complex( 0,1 ) * theta ) ) * spsigma,0,( 2 * ( np.e )**( np.complex( 0,-1 ) * theta ) * ( 2 * ( np.e )**( np.complex( 0,-1 ) * theta ) + 2 * ( np.e )**( np.complex( 0,1 ) * theta ) ) * ppsigma + 2 * ( np.e )**( np.complex( 0,1 ) * theta ) * ( 2 * ( np.e )**( np.complex( 0,-1 ) * theta ) + 2 * ( np.e )**( np.complex( 0,1 ) * theta ) ) * ppsigma ),0,-2 * ( 6 )**( 1/2 ) * ( np.e )**( ( np.complex( 0,1 ) * theta + np.complex( 0,2 ) * Phi ) ) * ( 2 * ( np.e )**( np.complex( 0,-1 ) * theta ) + 2 * ( np.e )**( np.complex( 0,1 ) * theta ) ) * pdsigma,0,( -6 * ( np.e )**( np.complex( 0,-1 ) * theta ) * ( 2 * ( np.e )**( np.complex( 0,-1 ) * theta ) + 2 * ( np.e )**( np.complex( 0,1 ) * theta ) ) * pdsigma + -6 * ( np.e )**( np.complex( 0,1 ) * theta ) * ( 2 * ( np.e )**( np.complex( 0,-1 ) * theta ) + 2 * ( np.e )**( np.complex( 0,1 ) * theta ) ) * pdsigma ),0,-2 * ( 6 )**( 1/2 ) * ( np.e )**( ( np.complex( 0,-1 ) * theta + np.complex( 0,-2 ) * Phi ) ) * ( 2 * ( np.e )**( np.complex( 0,-1 ) * theta ) + 2 * ( np.e )**( np.complex( 0,1 ) * theta ) ) * pdsigma,] ),np.array( [0,8 * ( np.e )**( np.complex( 0,2 ) * Phi ) * pppi,0,8 * pppi,0,0,0,0,0,] ),np.array( [2 * ( 6 )**( 1/2 ) * ( np.e )**( ( np.complex( 0,-1 ) * theta + np.complex( 0,-2 ) * Phi ) ) * sdsigma,0,( 4 * ( 6 )**( 1/2 ) * ( np.e )**( ( np.complex( 0,-2 ) * theta + np.complex( 0,-2 ) * Phi ) ) * pdsigma + 4 * ( 6 )**( 1/2 ) * ( np.e )**( np.complex( 0,-2 ) * Phi ) * pdsigma ),0,2 * ( np.e )**( ( np.complex( 0,1 ) * theta + np.complex( 0,2 ) * Phi ) ) * ( 8 * dddelta * ( np.e )**( ( np.complex( 0,-1 ) * theta + np.complex( 0,-2 ) * Phi ) ) + 12 * ddsigma * ( np.e )**( ( np.complex( 0,-1 ) * theta + np.complex( 0,-2 ) * Phi ) ) ),0,( 12 * ( 6 )**( 1/2 ) * ddsigma * ( np.e )**( ( np.complex( 0,-2 ) * theta + np.complex( 0,-2 ) * Phi ) ) +( 6 )**( 1/2 ) * ( np.e )**( np.complex( 0,1 ) * theta ) * ( 8 * dddelta * ( np.e )**( ( np.complex( 0,-1 ) * theta + np.complex( 0,-2 ) * Phi )) + 12 * ddsigma * ( np.e )**( ( np.complex( 0,-1 ) * theta + np.complex( 0,-2 ) * Phi ) ) ) ),0,24 * ddsigma * ( np.e )**( ( np.complex( 0,-2 ) * theta + np.complex( 0,-4 ) * Phi ) ),] ),np.array( [0,0,0,0,0,( np.e )**( np.complex( 0,1 ) * Phi ) * ( 4 * ( np.e )**( np.complex( 0,2) * theta ) * ( np.complex( 0,1 ) * ( 2 )**( -1/2 ) * ddpi * ( np.complex( 0,-3 ) * ( 2 )**( 1/2 ) * ( np.e )**( np.complex( 0,-1 ) * Phi ) + np.complex( 0,-1 ) * ( 2 )**( -1/2 ) * ( 4 * ( np.e )**( ( np.complex( 0,-2 ) * theta + np.complex( 0,-1 ) * Phi ) ) + 6 * ( np.e )**( np.complex(0,-1 ) * Phi ) ) ) + ( 2 )**( -1/2 ) * ddpi * ( -3 * ( 2 )**( 1/2 ) * ( np.e )**( np.complex( 0,-1 ) * Phi ) + ( 2 )**( -1/2 ) * ( 4 * ( np.e )**( ( np.complex( 0,-2 ) * theta + np.complex( 0,-1 ) * Phi ) ) + 6 * ( np.e )**( np.complex( 0,-1 ) * Phi ) ) ) ) + ( 6 )**( 1/2 ) *( ( 6 )**( 1/2 ) * ( np.complex( 0,1 ) * ( 2 )**( -1/2 ) * ddpi * ( np.complex( 0,-3 ) * ( 2 )**( 1/2 ) * ( np.e )**( np.complex( 0,-1 ) * Phi ) + np.complex( 0,-1 ) * ( 2 )**( -1/2 ) * ( 4 * ( np.e )**( ( np.complex( 0,-2 ) * theta + np.complex( 0,-1 ) * Phi ) ) + 6 * ( np.e )**( np.complex( 0,-1 ) * Phi ) ) ) + -1 * ( 2 )**( -1/2 ) * ddpi * ( -3 * ( 2 )**( 1/2 ) * ( np.e )**( np.complex( 0,-1 ) * Phi ) + ( 2 )**( -1/2 ) * ( 4 * ( np.e )**( ( np.complex( 0,-2 ) * theta + np.complex(0,-1 ) * Phi ) ) + 6 * ( np.e )**( np.complex( 0,-1 ) * Phi ) ) ) ) + ( 6 )**( 1/2 ) * ( np.complex( 0,1 ) * ( 2 )**( -1/2 ) * ddpi * ( np.complex( 0,-3 ) * ( 2 )**( 1/2 ) * ( np.e )**( np.complex( 0,-1 ) * Phi ) + np.complex( 0,-1 ) * ( 2 )**( -1/2 ) * ( 4 * ( np.e )**( ( np.complex( 0,-2 ) * theta + np.complex( 0,-1 ) * Phi ) ) + 6 * ( np.e )**( np.complex( 0,-1 ) * Phi ) ) ) + ( 2 )**( -1/2 ) * ddpi * ( -3 *( 2 )**( 1/2 ) * ( np.e )**( np.complex( 0,-1 ) * Phi ) + ( 2 )**( -1/2 ) * ( 4 * ( np.e )**( ( np.complex( 0,-2 ) * theta + np.complex( 0,-1) * Phi ) ) + 6 * ( np.e )**( np.complex( 0,-1 ) * Phi ) ) ) ) ) ),0,( np.e )**( np.complex( 0,-1 ) * Phi ) * ( 4 * ( np.e )**( np.complex( 0,-2 ) * theta ) * ( np.complex( 0,1 ) * ( 2 )**( -1/2 ) * ddpi * ( np.complex( 0,-3 ) * ( 2 )**( 1/2 ) * ( np.e )**( np.complex( 0,-1 ) *Phi ) + np.complex( 0,-1 ) * ( 2 )**( -1/2 ) * ( 4 * ( np.e )**( ( np.complex( 0,-2 ) * theta + np.complex( 0,-1 ) * Phi ) ) + 6 * ( np.e )**( np.complex( 0,-1 ) * Phi ) ) ) + -1 * ( 2 )**( -1/2 ) * ddpi * ( -3 * ( 2 )**( 1/2 ) * ( np.e )**( np.complex( 0,-1 ) * Phi ) + ( 2 )**( -1/2 ) * ( 4 * ( np.e )**( ( np.complex( 0,-2 ) * theta + np.complex( 0,-1 ) * Phi ) ) + 6 * ( np.e )**( np.complex( 0,-1 ) * Phi ) ) ) ) + ( 6 )**( 1/2 ) * ( ( 6 )**( 1/2 ) * ( np.complex( 0,1 ) * ( 2 )**( -1/2 ) * ddpi * ( np.complex( 0,-3 ) * ( 2 )**( 1/2 ) * ( np.e )**( np.complex( 0,-1 ) * Phi ) + np.complex( 0,-1 ) * ( 2 )**( -1/2 ) * ( 4 * ( np.e )**( ( np.complex( 0,-2 ) * theta + np.complex( 0,-1 ) * Phi )) + 6 * ( np.e )**( np.complex( 0,-1 ) * Phi ) ) ) + -1 * ( 2 )**( -1/2 ) * ddpi * ( -3 * ( 2 )**( 1/2 ) * ( np.e )**( np.complex( 0,-1 ) *Phi ) + ( 2 )**( -1/2 ) * ( 4 * ( np.e )**( ( np.complex( 0,-2 ) * theta + np.complex( 0,-1 ) * Phi ) ) + 6 * ( np.e )**( np.complex( 0,-1 ) * Phi ) ) ) ) + ( 6 )**( 1/2 ) * ( np.complex( 0,1 ) * ( 2 )**( -1/2 ) * ddpi * ( np.complex( 0,-3 ) * ( 2 )**( 1/2 ) * ( np.e )**( np.complex( 0,-1 ) * Phi ) + np.complex( 0,-1 ) * ( 2 )**( -1/2 ) * ( 4* ( np.e )**( ( np.complex( 0,-2 ) * theta + np.complex( 0,-1 ) * Phi )) + 6 * ( np.e )**( np.complex( 0,-1 ) * Phi ) ) ) + ( 2 )**( -1/2 ) * ddpi * ( -3 * ( 2 )**( 1/2 ) * ( np.e )**( np.complex( 0,-1 ) * Phi ) + ( 2 )**( -1/2 ) * ( 4 * ( np.e )**( ( np.complex( 0,-2 ) * theta +np.complex( 0,-1 ) * Phi ) ) + 6 * ( np.e )**( np.complex( 0,-1 ) * Phi ) ) ) ) ) ),0,] ),np.array( [( 6 * ( np.e )**( np.complex( 0,-1 ) * theta ) + 6 * ( np.e )**( np.complex( 0,1 ) * theta ) ) * sdsigma,0,(2 * ( np.e )**( np.complex( 0,-1 ) * theta ) * ( 6 * ( np.e )**( np.complex( 0,-1 ) * theta ) + 6 * ( np.e )**( np.complex( 0,1 ) * theta )) * pdsigma + 2 * ( np.e )**( np.complex( 0,1 ) * theta ) * ( 6 * ( np.e )**( np.complex( 0,-1 ) * theta ) + 6 * ( np.e )**( np.complex( 0,1 ) * theta ) ) * pdsigma ),0,2 * ( np.e )**( ( np.complex( 0,1 ) * theta +np.complex( 0,2 ) * Phi ) ) * ( ( 6 )**( 1/2 ) * ddsigma * ( 6 * ( np.e )**( np.complex( 0,-1 ) * theta ) + 6 * ( np.e )**( np.complex( 0,1 ) * theta ) ) + 2 * ( np.complex( 0,1 ) * ( 2 )**( -1/2 ) * dddelta * (np.complex( 0,-2 ) * ( 3 )**( 1/2 ) * ( np.e )**( np.complex( 0,-1 ) * theta ) + np.complex( 0,2 ) * ( 3 )**( 1/2 ) * ( np.e )**( np.complex( 0,1) * theta ) ) + ( 2 )**( -1/2 ) * dddelta * ( 2 * ( 3 )**( 1/2 ) * ( np.e )**( np.complex( 0,-1 ) * theta ) + 2 * ( 3 )**( 1/2 ) * ( np.e )**( np.complex( 0,1 ) * theta ) ) ) ),0,( ( 6 )**( 1/2 ) * ( np.e )**(np.complex( 0,-1 ) * theta ) * ( ( 6 )**( 1/2 ) * ddsigma * ( 6 * ( np.e)**( np.complex( 0,-1 ) * theta ) + 6 * ( np.e )**( np.complex( 0,1 ) * theta ) ) + 2 * ( np.complex( 0,-1 ) * ( 2 )**( -1/2 ) * dddelta * ( np.complex( 0,-2 ) * ( 3 )**( 1/2 ) * ( np.e )**( np.complex( 0,-1 ) * theta ) + np.complex( 0,2 ) * ( 3 )**( 1/2 ) * ( np.e )**( np.complex( 0,1) * theta ) ) + ( 2 )**( -1/2 ) * dddelta * ( 2 * ( 3 )**( 1/2 ) * ( np.e )**( np.complex( 0,-1 ) * theta ) + 2 * ( 3 )**( 1/2 ) * ( np.e )**( np.complex( 0,1 ) * theta ) ) ) ) + ( 6 )**( 1/2 ) * ( np.e )**( np.complex( 0,1 ) * theta ) * ( ( 6 )**( 1/2 ) * ddsigma * ( 6 * ( np.e )**( np.complex( 0,-1 ) * theta ) + 6 * ( np.e )**( np.complex( 0,1 ) * theta ) ) + 2 * ( np.complex( 0,1 ) * ( 2 )**( -1/2 ) * dddelta * ( np.complex( 0,-2 ) * ( 3 )**( 1/2 ) * ( np.e )**( np.complex( 0,-1 ) * theta ) + np.complex( 0,2 ) * ( 3 )**( 1/2 ) * ( np.e )**( np.complex( 0,1) * theta ) ) + ( 2 )**( -1/2 ) * dddelta * ( 2 * ( 3 )**( 1/2 ) * ( np.e )**( np.complex( 0,-1 ) * theta ) + 2 * ( 3 )**( 1/2 ) * ( np.e )**( np.complex( 0,1 ) * theta ) ) ) ) ),0,2 * ( np.e )**( ( np.complex(0,-1 ) * theta + np.complex( 0,-2 ) * Phi ) ) * ( ( 6 )**( 1/2 ) * ddsigma * ( 6 * ( np.e )**( np.complex( 0,-1 ) * theta ) + 6 * ( np.e )**( np.complex( 0,1 ) * theta ) ) + 2 * ( np.complex( 0,-1 ) * ( 2 )**(-1/2 ) * dddelta * ( np.complex( 0,-2 ) * ( 3 )**( 1/2 ) * ( np.e )**(np.complex( 0,-1 ) * theta ) + np.complex( 0,2 ) * ( 3 )**( 1/2 ) * ( np.e)**( np.complex( 0,1 ) * theta ) ) + ( 2 )**( -1/2 ) * dddelta * ( 2 * ( 3 )**( 1/2 ) * ( np.e )**( np.complex( 0,-1 ) * theta ) + 2 * ( 3 )**( 1/2 ) * ( np.e )**( np.complex( 0,1 ) * theta ) ) ) ),] ),np.array( [0,0,0,0,0,( np.e )**( np.complex( 0,1 ) * Phi ) * ( 4 * ( np.e )**( np.complex( 0,2 ) * theta ) * ( ( 2 )**( -1/2 ) * ddpi * ( 3 * ( 2 )**( 1/2 ) * ( np.e )**( np.complex( 0,1 ) * Phi ) + -1 * ( 2 )**( -1/2 ) * ( 4 * ( np.e )**( ( np.complex( 0,2 ) * theta + np.complex( 0,1 ) * Phi ) ) + 6 * ( np.e )**( np.complex( 0,1 ) * Phi ) ) ) + np.complex(0,1 ) * ( 2 )**( -1/2 ) * ddpi * ( np.complex( 0,-3 ) * ( 2 )**( 1/2 ) * ( np.e )**( np.complex( 0,1 ) * Phi ) + np.complex( 0,-1 ) * ( 2 )**(-1/2 ) * ( 4 * ( np.e )**( ( np.complex( 0,2 ) * theta + np.complex( 0,1 )* Phi ) ) + 6 * ( np.e )**( np.complex( 0,1 ) * Phi ) ) ) ) + ( 6 )**( 1/2 ) * ( ( 6 )**( 1/2 ) * ( -1 * ( 2 )**( -1/2 ) * ddpi * ( 3 * ( 2 )**( 1/2 ) * ( np.e )**( np.complex( 0,1 ) * Phi ) + -1 * ( 2 )**( -1/2 ) * ( 4 * ( np.e )**( ( np.complex( 0,2 ) * theta + np.complex( 0,1 ) * Phi ) ) + 6 * ( np.e )**( np.complex( 0,1 ) * Phi ) ) ) + np.complex( 0,1 ) * ( 2 )**( -1/2 ) * ddpi * ( np.complex( 0,-3 ) * ( 2 )**( 1/2 ) * ( np.e )**( np.complex( 0,1 ) * Phi ) + np.complex( 0,-1 )* ( 2 )**( -1/2 ) * ( 4 * ( np.e )**( ( np.complex( 0,2 ) * theta + np.complex( 0,1 ) * Phi ) ) + 6 * ( np.e )**( np.complex( 0,1 ) * Phi ) ) ) ) + ( 6 )**( 1/2 ) * ( ( 2 )**( -1/2 ) * ddpi * ( 3 * ( 2 )**( 1/2 ) * ( np.e )**( np.complex( 0,1 ) * Phi ) + -1 * ( 2 )**( -1/2 ) * ( 4 * ( np.e )**( ( np.complex( 0,2 ) * theta + np.complex( 0,1 ) * Phi ) ) + 6 * ( np.e )**( np.complex( 0,1 ) * Phi ) ) ) + np.complex(0,1 ) * ( 2 )**( -1/2 ) * ddpi * ( np.complex( 0,-3 ) * ( 2 )**( 1/2 ) * ( np.e )**( np.complex( 0,1 ) * Phi ) + np.complex( 0,-1 ) * ( 2 )**(-1/2 ) * ( 4 * ( np.e )**( ( np.complex( 0,2 ) * theta + np.complex( 0,1 ) * Phi ) ) + 6 * ( np.e )**( np.complex( 0,1 ) * Phi ) ) ) ) ) ),0,( np.e )**( np.complex( 0,-1 ) * Phi ) * ( 4 * ( np.e )**( np.complex( 0,-2 ) * theta ) * ( -1 * ( 2 )**( -1/2 ) * ddpi * ( 3 * ( 2 )**( 1/2 ) * ( np.e )**( np.complex( 0,1 ) * Phi ) + -1 * ( 2 )**( -1/2 ) * ( 4 * ( np.e )**( ( np.complex( 0,2 ) * theta + np.complex( 0,1 ) * Phi ) ) + 6 * ( np.e )**( np.complex( 0,1 ) * Phi ) ) ) + np.complex( 0,1 ) * ( 2 )**( -1/2 ) * ddpi * ( np.complex( 0,-3 ) * ( 2 )**( 1/2 ) * ( np.e )**( np.complex( 0,1 ) * Phi ) + np.complex( 0,-1 ) * ( 2 )**( -1/2 ) * ( 4 * ( np.e )**( ( np.complex( 0,2 ) * theta +np.complex( 0,1 ) * Phi ) ) + 6 * ( np.e )**( np.complex( 0,1 ) * Phi ) ) ) ) + ( 6 )**( 1/2 ) * ( ( 6 )**( 1/2 ) * ( -1 * ( 2 )**( -1/2 ) * ddpi * ( 3 * ( 2 )**( 1/2 ) * ( np.e )**( np.complex( 0,1 ) * Phi ) + -1 * ( 2 )**( -1/2 ) * ( 4 * ( np.e )**( ( np.complex( 0,2 ) * theta + np.complex( 0,1 ) * Phi ) ) + 6 * ( np.e )**( np.complex( 0,1 ) * Phi ) ) ) + np.complex( 0,1 ) * ( 2 )**( -1/2 ) * ddpi * ( np.complex( 0,-3 ) *( 2 )**( 1/2 ) * ( np.e )**( np.complex( 0,1 ) * Phi ) + np.complex( 0,-1 ) * ( 2 )**( -1/2 ) * ( 4 * ( np.e )**( ( np.complex( 0,2 ) * theta + np.complex( 0,1 ) * Phi ) ) + 6 * ( np.e )**( np.complex( 0,1 ) * Phi ) ) ) ) + ( 6 )**( 1/2 ) * ( ( 2 )**( -1/2 ) * ddpi * ( 3 * ( 2 )**( 1/2 ) * ( np.e )**( np.complex( 0,1 ) * Phi ) + -1 * ( 2 )**(-1/2 ) * ( 4 * ( np.e )**( ( np.complex( 0,2 ) * theta + np.complex( 0,1 ) * Phi ) ) + 6 * ( np.e )**( np.complex( 0,1 ) * Phi ) ) ) + np.complex( 0,1 ) * ( 2 )**( -1/2 ) * ddpi * ( np.complex( 0,-3 ) * ( 2 )**( 1/2 ) * ( np.e )**( np.complex( 0,1 ) * Phi ) + np.complex( 0,-1 ) * ( 2 )**( -1/2 ) * ( 4 * ( np.e )**( ( np.complex( 0,2 ) * theta + np.complex( 0,1 ) * Phi ) ) + 6 * ( np.e )**( np.complex( 0,1 ) * Phi ) ) ) ) ) ),0,] ),np.array( [2 * ( 6 )**( 1/2 ) * ( np.e )**( ( np.complex( 0,1 ) * theta + np.complex( 0,2 ) * Phi ) ) * sdsigma,0,( 4 * ( 6 )**( 1/2 ) * ( np.e )**( ( np.complex( 0,2 ) * theta + np.complex( 0,2 ) * Phi ) ) * pdsigma + 4 * ( 6 )**( 1/2 ) * ( np.e )**( np.complex( 0,2 ) * Phi ) * pdsigma ),0,24 * ddsigma * ( np.e )**( ( np.complex( 0,2 ) * theta + np.complex( 0,4 ) * Phi ) ),0,( 12 * ( 6 )**( 1/2 ) * ddsigma * ( np.e )**( ( np.complex( 0,2 ) * theta + np.complex( 0,2 ) * Phi ) ) + ( 6 )**( 1/2 ) * ( np.e )**( np.complex(0,-1 ) * theta ) * ( 8 * dddelta * ( np.e )**( ( np.complex( 0,1 ) * theta + np.complex( 0,2 ) * Phi ) ) + 12 * ddsigma * ( np.e )**( ( np.complex( 0,1 ) * theta + np.complex( 0,2 ) * Phi ) ) ) ),0,2 * ( np.e )**( ( np.complex( 0,-1 ) * theta + np.complex( 0,-2 ) * Phi ) ) * ( 8 * dddelta * ( np.e )**( ( np.complex( 0,1 ) * theta + np.complex( 0,2 ) * Phi ) ) + 12 * ddsigma * ( np.e )**( ( np.complex( 0,1 ) * theta + np.complex( 0,2 ) * Phi ) ) ),] ),] )
    eigvals = np.linalg.eigvalsh(hamiltonian)
    eigvals.sort()
    return eigvals

def band_structure(params, neighbors, path):
    
    bands = []
    
    for k in np.vstack(path):
        g = phase(k, neighbors)
        print(g.shape)
        eigvals = band_energies(*params)
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

# sssigma , spsigma , sdsigma , ppsigma , pppi , pdsigma , ddsigma , ddpi , dddelta
sssigma = -0.07835; spsigma = 0.11192; sdsigma = -0.06197; ppsigma = 0.18677; pppi = -0.02465; 
pdsigma = -0.08564; ddsigma = -0.06856; ddpi = 0.03528; dddelta = -0.00588; pdpi = 0.02446;
params = (sssigma , spsigma , sdsigma , ppsigma , pppi , pdsigma , ddsigma , ddpi , dddelta, pdpi)

# k-points per path
n = 1000

# lattice constant
a = 1

# nearest neighbors to atom at (0, 0, 0)
#https://arxiv.org/pdf/1004.2974.pdf
neighbors = a / 12 *  np.array([
    [0, 1, 1], 
    [0, -1, -1], 
    [1, 0, 1], 
    [-1, 0, -1],
    [1, 1, 0],
    [-1, -1, 0],
    [0, 1, 1],
    [0, 1, -1],
    [1, 0, -1],
    [-1, 0, 1],
    [-1, 1, 0],
    [1, -1, 0]])

# symmetry points in the Brillouin zone
G = 2 * np.pi / a * np.array([0, 0, 0])
L = 2 * np.pi / a * np.array([1/2, 1/2, 1/2])
K = 2 * np.pi / a * np.array([3/8, 3/8, 3/8])
X = 2 * np.pi / a * np.array([1/2, 0, 1/2])
W = 2 * np.pi / a * np.array([1/2, 1/4, 3/4])
U = 2 * np.pi / a * np.array([5/8, 1/4, 5/8])

# k-paths
lambd = linpath(L, G, n, endpoint=False)
delta = linpath(G, X, n, endpoint=False)
x_uk = linpath(X, U, n // 4, endpoint=False)
sigma = linpath(K, G, n, endpoint=True)

# MAIN
theta=2.0;
Phi=50.0;
bands = band_structure(params, neighbors, path=[lambd, delta, x_uk, sigma])
# be = band_energies( sssigma , spsigma , sdsigma , ppsigma , pppi , pdsigma , ddsigma , ddpi , dddelta, pdpi)

# print(be)



#matplotlib inline

plt.figure(figsize=(15, 9))

ax = plt.subplot(111)

# remove plot borders
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

# limit plot area to data
plt.xlim(0, len(bands))
plt.ylim(min(bands[0]) - 1, max(bands[7]) + 1)

# custom tick names for k-points
xticks = n * np.array([0, 0.5, 1, 1.5, 2, 2.25, 2.75, 3.25])
plt.xticks(xticks, ('$L$', '$\Lambda$', '$\Gamma$', '$\Delta$', '$X$', '$U,K$', '$\Sigma$', '$\Gamma$'), fontsize=18)
plt.yticks(fontsize=18)

# horizontal guide lines every 2.5 eV
for y in np.arange(-25, 25, 2.5):
    plt.axhline(y, ls='--', lw=0.3, color='black', alpha=0.3)

# hide ticks, unnecessary with gridlines
plt.tick_params(axis='both', which='both',
                top='off', bottom='off', left='off', right='off',
                labelbottom='on', labelleft='on', pad=5)

plt.xlabel('k-Path', fontsize=20)
plt.ylabel('Energy (eV)', fontsize=20)

plt.text(1350, -18, 'Fig. 1. Band structure of Si.', fontsize=12)

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

plt.show()