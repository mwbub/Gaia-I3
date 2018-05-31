"""
NAME:
    User_Interaction

PURPOSE:
    Contain functions that allow for communication with user efficiently
    
HISTORY:
    2018-05-30 - Written - Samuel Wong
"""
from Frame_Conversion import *

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