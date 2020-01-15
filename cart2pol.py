import numpy as np

x = -float(input('xvalue: '))
y = float(input('yvalue: '))

r = np.sqrt(x**2 + y**2)
theta = np.arctan2(y,x) * 180 / np.pi

#print(r,theta)

if theta>0 and theta<90:
    theta_new = theta+270
if theta>90 and theta<360:
    theta_new = theta-90
if theta<0:
    theta_new = 270+theta

print(r,theta_new)
