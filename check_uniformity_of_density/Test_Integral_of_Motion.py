from Integral_of_Motion import *

#global variables
# for the position, I took an example from online, the answer is (2, pi/6, 4)
x = np.sqrt(3)
y = 1.
z = 4.
# for the velocity, I chose the test to see whether the total velocity is the same, which is clealry 3 here
vx = np.cbrt(3)
vy = np.cbrt(3)
vz = np.cbrt(3)


def test_cartesian_to_cylindrical(x, y, z, vx, vy, vz):
    return cartesian_to_cylindrical(x, y, z, vx, vy, vz)



