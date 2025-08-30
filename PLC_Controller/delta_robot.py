import math
# Trigonometric constants
sqrt3 = math.sqrt(3.0)
pi = 3.141592653
sin120 = sqrt3 / 2.0
cos120 = -0.5
tan60 = sqrt3
sin30 = 0.5
tan30 = 1 / sqrt3
# Robot geometry (mm)
Up =50  
Wb =150  # base radius
f=Wb*6/sqrt3
e=Up*3/sqrt3  
re = 473  # end to elbow distance
rf =250   # base to elbow distance



# Forward kinematics: (theta1, theta2, theta3) -> (x0, y0, z0) in mm
# Returned status: 0=OK, -1=non-existing position
def delta_calcForward(theta1, theta2, theta3):
    t = (Wb - Up) 
    dtr = pi / 180.0
    
    theta1 *= dtr
    theta2 *= dtr
    theta3 *= dtr
    
    y1 = -(t + rf * math.cos(theta1))
    z1 = -rf * math.sin(theta1)
    
    y2 = (t + rf * math.cos(theta2)) * sin30
    x2 = y2 * tan60
    z2 = -rf * math.sin(theta2)
    
    y3 = (t + rf * math.cos(theta3)) * sin30
    x3 = -y3 * tan60
    z3 = -rf * math.sin(theta3)
    
    dnm = (y2 - y1) * x3 - (y3 - y1) * x2
    
    w1 = y1 * y1 + z1 * z1
    w2 = x2 * x2 + y2 * y2 + z2 * z2
    w3 = x3 * x3 + y3 * y3 + z3 * z3
    
    # x = (a1*z + b1)/dnm
    a1 = (z2 - z1) * (y3 - y1) - (z3 - z1) * (y2 - y1)
    b1 = -((w2 - w1) * (y3 - y1) - (w3 - w1) * (y2 - y1)) / 2.0
    
    # y = (a2*z + b2)/dnm
    a2 = -(z2 - z1) * x3 + (z3 - z1) * x2
    b2 = ((w2 - w1) * x3 - (w3 - w1) * x2) / 2.0
    
    # a*z^2 + b*z + c = 0
    a = a1 * a1 + a2 * a2 + dnm * dnm
    b = 2 * (a1 * b1 + a2 * (b2 - y1 * dnm) - z1 * dnm * dnm)
    c = (b2 - y1 * dnm) * (b2 - y1 * dnm) + b1 * b1 + dnm * dnm * (z1 * z1 - re * re)
    
    # Discriminant
    d = b * b - 4.0 * a * c
    if d < 0:
        return (-1, 0, 0, 0)  # non-existing point
    
    z0 = -0.5 * (b + math.sqrt(d)) / a
    x0 = (a1 * z0 + b1) / dnm
    y0 = (a2 * z0 + b2) / dnm
    return (0, x0, y0, z0)

# Inverse kinematics helper function, calculates angle theta1 (for YZ-plane) in degrees
def delta_calcAngleYZ(x0, y0, z0):
    y1 = -0.5 * 0.57735 * f  # f/2 * tan(30)
    y0 -= Up  # shift center to edge
    # z = a + b*y
    #a = (x0 * x0 + y0 * y0 + z0 * z0 + rf * rf - re * re - y1 * y1) / (2 * z0)
    a = ( x0 * x0+ y0 * y0 + z0 * z0 + rf * rf - re * re - y1 * y1) / (2 * z0)
    b = (y1 - y0) / z0
    # Discriminant
    d = -(a + b * y1) * (a + b * y1) + rf * (b * b * rf + rf)
    if d < 0:
        return (-1, 0)  # non-existing point
    yj = (y1 - a * b - math.sqrt(d)) / (b * b + 1)  # choosing outer point
    zj = a + b * yj
    if yj > y1:
        theta = 180.0 * math.atan(-zj / (y1 - yj)) / pi + 180.0
    else:
        theta = 180.0 * math.atan(-zj / (y1 - yj)) / pi
    return (0, theta)

# Inverse kinematics: (x0, y0, z0) -> (theta1, theta2, theta3) in degrees
# Returned status: 0=OK, -1=non-existing position
def delta_calcInverse(x0, y0, z0):
    theta1 = theta2 = theta3 = 0.0
    status, theta1 = delta_calcAngleYZ(x0, y0, z0)
    
    if status == 0:
        status, theta2 = delta_calcAngleYZ(x0 * cos120 + y0 * sin120, y0 * cos120 - x0 * sin120, z0)  # rotate coords to +120 deg
    
    if status == 0:
        status, theta3 = delta_calcAngleYZ(x0 * cos120 - y0 * sin120, y0 * cos120 + x0 * sin120, z0)  # rotate coords to -120 deg
    
    return (status, theta1, theta2, theta3)

# Test the functions
if __name__ == "__main__":
    # Test inverse kinematics
    x, y, z = 0, 0, -500  # Position 
    status, theta1, theta2, theta3 = delta_calcInverse(x, y, z)
    if status == 0:
        print(f"Inverse Kinematics: theta1 = {theta1:.2f}°, theta2 = {theta2:.2f}°, theta3 = {theta3:.2f}°")
    else:
        print("Inverse Kinematics: No solution exists")
    
    # Test forward kinematics
    if status == 0:
        status, x0, y0, z0 = delta_calcForward(theta1, theta2, theta3)
        if status == 0:
            print(f"Forward Kinematics: x = {x0:.2f} mm, y = {y0:.2f} mm, z = {z0:.2f} mm")
        else:
            print("Forward Kinematics: No solution exists")