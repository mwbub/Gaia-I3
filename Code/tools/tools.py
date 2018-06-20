"""
NAME:
    tools

PURPOSE:
    Contains miscellaneous tools, including frame conversion, interaction with
    user, unit conversion.
    
HISTORY:
    2018-05-31 - Written - Samuel Wong
    2018-06-19 - Added Amount of Standard Deviation Cut function - Michael Poon
"""
import numpy as np
from astropy.coordinates import SkyCoord, CartesianRepresentation, CartesianDifferential
import astropy.units as unit

def galactic_to_galactocentric(point):
    """
    NAME:
        galactic_to_galactocentric

    PURPOSE:
        Given 6 coordinates for the position and velocity of a star in
        galactic Cartesian coordinate, convert to galactocentric Cartesian

    INPUT:
        point = numpy array ([u, v, w, U, V, W]), where:
            u =  u coordinate in kpc
            v = v coordinate in kpc
            w = w coordinate in kpc
            U = velocity in U in km/s
            V = velocity in V in km/s
            W = velocity in W in km/s

    OUTPUT:
        numpy array ([x, y, z, vx, vy, vz]), where:
            x =  x coordinate in kpc
            y = y coordinate in kpc
            z = z coordinate in kpc
            vx = velocity in x in km/s
            vy = velocity in y in km/s
            vz = velocity in z in km/s

    HISTORY:
        2018-05-30 - Written - Samuel Wong
    """
    u, v, w, U, V, W = point
    coord = SkyCoord(frame = 'galactic', representation_type = CartesianRepresentation,
                 differential_type = CartesianDifferential,
                 u = u*unit.kpc, v = v*unit.kpc, w = w*unit.kpc,
                 U = U*unit.km/unit.s, V = V*unit.km/unit.s, W = W*unit.km/unit.s)
    coord = coord.transform_to('galactocentric')
    coord.representation_type = CartesianRepresentation
    x = coord.x.value
    y = coord.y.value
    z = coord.z.value
    vx = coord.v_x.value
    vy = coord.v_y.value
    vz = coord.v_z.value
    return np.array([x, y, z, vx, vy, vz])


def galactocentric_to_galactic(point):
    """
    NAME:
        galactocentric_to_galactic

    PURPOSE:
        Given 6 coordinates for the position and velocity of a star in
        galactocentric Cartesian coordinate, convert to galactic Cartesian

    INPUT:
        point = numpy array ([x, y, z, vx, vy, vz]), where:
            x =  x coordinate in kpc
            y = y coordinate in kpc
            z = z coordinate in kpc
            vx = velocity in x in km/s
            vy = velocity in y in km/s
            vz = velocity in z in km/s

    OUTPUT:            
        numpy array ([u, v, w, U, V, W]), where:
            u =  u coordinate in kpc
            v = v coordinate in kpc
            w = w coordinate in kpc
            U = velocity in U in km/s
            V = velocity in V in km/s
            W = velocity in W in km/s

    HISTORY:
        2018-05-30 - Written - Samuel Wong
    """
    x, y, z, vx, vy, vz = point
    coord = SkyCoord(frame = 'galactocentric', representation_type = CartesianRepresentation,
                 differential_type = CartesianDifferential,
                 x = x*unit.kpc, y = y*unit.kpc, z = z*unit.kpc,
                 v_x = vx*unit.km/unit.s, v_y = vy*unit.km/unit.s,
                 v_z = vz*unit.km/unit.s)
    coord = coord.transform_to('galactic')
    coord.representation_type = CartesianRepresentation
    u = coord.u.value
    v = coord.v.value
    w = coord.w.value
    U = coord.U.value
    V = coord.V.value
    W = coord.W.value
    return np.array([u, v, w, U, V, W])


def to_natural_units(list_of_coord):
    """
    NAME:
        to_natural_units

    PURPOSE:
        Given a list of coordinates, return a list of the same shape but
        changed to natural units by taking the ratio with the Sun's positio
        and velocity
        
    INPUT:
        list_of_coord = an array of coordinates, each coordinate has 6 components
            of positions and velocities, in cartesian galactocentric.
            Assumed to be in unit of kpc and km/s

    OUTPUT:            
        natural_list = an array of coordinates, each coordinate has 6 components
            of positions and velocities, in natural units

    HISTORY:
        2018-05-30 - Written - Samuel Wong
    """

    natural_list = np.empty(np.shape(list_of_coord))
    for (i, coord) in enumerate(list_of_coord):
        x, y, z, vx, vy, vz = coord
        natural_list[i] = np.array([x/8., y/8., z/8., vx/220., vy/220., vz/220.])
    return natural_list
    

def get_star_coord_from_user():
    """
    NAME:
        cartesian_to_cylindrical

    PURPOSE:
        
    Input: 
        User is prompted to input coordinate of stars

    OUTPUT:
        (point_galactocentric, point_galactic) where each component is a 
        numpy array, which contains 6 numbers, 3 position and 3 velocity, in 
        galactocentric and galactic, respectively.
        The numbers are assumed to be in unit of kpc and km/s.

    HISTORY:
        2018-05-30 - Written - Samuel Wong
    """
    # initialize repeat boolean
    repeat = True
    
    while (repeat):
        # ask the user for input coordinate frame
        frame = input("Do you want to search star in galactic or galactocentric coordinate? ")
        if frame == "galactic":
            repeat = False
            print("Please enter position in kpc and velocity in km/s.")
            u  = float(input('u = '))
            v  = float(input('v = '))
            w  = float(input('w = '))
            U  = float(input('U = '))
            V  = float(input('V = '))
            W  = float(input('W = '))
            point_galactic = np.array([u, v, w, U, V, W])
            point_galactocentric = galactic_to_galactocentric(point_galactic)
        elif frame == "galactocentric":
            repeat = False
            print("Please enter position in kpc and velocity in km/s.")
            x  = float(input('x = '))
            y  = float(input('y = '))
            z  = float(input('z = '))
            vx  = float(input('vx = '))
            vy = float(input('vy = '))
            vz  = float(input('vz = '))
            point_galactocentric = np.array([x, y, z, vx, vy, vz])
            point_galactic = galactocentric_to_galactic(point_galactocentric)
            
    return (point_galactocentric, point_galactic)


def create_meshgrid(xy_min, xy_max, xy_spacing, z_min, z_max, z_spacing,
                    vxy_min, vxy_max, vxy_spacing, vz_min, vz_max, vz_spacing):
    """
    NAME:
        create_meshgrid

    PURPOSE:
        Create a meshgrid of R^6 values used to evaluate many points easily along
        a given density function.

    Input:
        User is prompted to input the minimum, maximum and spacing values of each
        galactocentric coordinate (x, y, z, vx, vy, vz). The meshgrid goes from minimum
        to maximum inclusive.

        Note: xy are put together to simplify input as we assume the bounds and spacing should
        be the same. We assume the circle of Milky Way stars is symmetric in x, y, vx and vy.
        vxy_min and vxy_max refers to the bounds of vx and vy as the same.

    OUTPUT:
        numpy array of 6 dimensional tuples, in the order of (x, y, z, vx, vy, vy)
        x, y and x are in kpc.
        vx, vy and vz are in km/s.
        We artifically remove any point whose (x,y,z) = (0,0,0)

    HISTORY:
        2018-06-04 - Written - Michael Poon
    """

    x = np.arange(xy_min, xy_max + xy_spacing, xy_spacing)
    y = np.arange(xy_min, xy_max + xy_spacing, xy_spacing)
    z = np.arange(z_min, z_max + z_spacing, z_spacing)
    vx = np.arange(vxy_min, vxy_max + vxy_spacing, vxy_spacing)
    vy = np.arange(vxy_min, vxy_max + vxy_spacing, vxy_spacing)
    vz = np.arange(vz_min, vz_max + vz_spacing, vz_spacing)

    x_values, y_values, z_values, vx_values, vy_values, vz_values = np.meshgrid(x, y, z, vx, vy, vz)

    x_values = x_values.reshape(-1, 1)  # -1 is flatten and 1 is put into 1D column
    y_values = y_values.reshape(-1, 1)
    z_values = z_values.reshape(-1, 1)
    vx_values = vx_values.reshape(-1, 1)
    vy_values = vy_values.reshape(-1, 1)
    vz_values = vz_values.reshape(-1, 1)

    meshgrid = np.concatenate((x_values, y_values, z_values, vx_values, vy_values, vz_values), axis=1)
    
    return meshgrid


def std_cut(samples, number_of_std_cut):
    
    """
    NAME:
        std_cut
    
    PURPOSE:
        Remove outliers (points with any coordinate value further 
        than a multiple of std from the mean) as a quality cut.
        
    INPUT:
        1. Nx6 array of rectangular phase space coordinates of the form 
        (x, y, z, vx, vy, vz) in [kpc, kpc, kpc, km/s, km/s, km/s].
        
        2. Amount of standard deviation cut from each coordinate
        
    OUTPUT:
        Nx6 array of rectangular phase space coordinates of the form 
        (x, y, z, vx, vy, vz) in [kpc, kpc, kpc, km/s, km/s, km/s].
    
    NOTES:
        From a sample of 6376803 (20% parallax cut), if we do a cut of 
        (3.0 std) -> new_sample = 5950618 (93.3% remaining / 426185 cut)
        (2.5 std) -> new_sample = 5658852 (88.7% remaining / 717951 cut)
        (2.0 std) -> new_sample = 5131168 (80.4% remaining / 1245635 cut)
        
    HISTORY:
        2018-06-04 - Written - Michael Poon
    """
    
    print("With", len(samples), "samples, now performing a", number_of_std_cut, "std quality cut.")
    
    mean, std = np.mean(samples, axis=0), np.std(samples, axis=0, ddof=1)
    new_samples = samples[np.all(np.abs((samples - mean) / std) < number_of_std_cut, axis=1)]
    print(len(samples) - len(new_samples), "samples cut. Now there are", len(new_samples), "samples.")
    return new_samples    
