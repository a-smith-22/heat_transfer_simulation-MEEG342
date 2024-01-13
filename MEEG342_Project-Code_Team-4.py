'''
TITLE: Explicit Finite Difference Heat Transfer Solver
AUTHOR(s): Andrew Smith, Noah Etienne
DATE: Apr 30 2023 (last updated)
NOTES: CTRL+L to clear terminal, F5 to run. 

UPDATES:
-V1.1 (4/19): Rewrote code to use max Fo number, input parameter only n_cells value (line 35)
-V1.2 (4/19): Added input conditions and improved user interface. Data is saved to CSV file for analysis.
-V1.2.1 (4/19): Fixed graph output title. Converted temperatures to Celsius. Added plot interpolation control.
-V1.2.2 (4/19): Added time module to checked computational wall time for finite difference section.
-V1.3 (4/19): Added expected computational time.
-V1.4 (4/22): Embedded finite difference equations in main loop for faster computation. Fixed error in property values. 
-V1.4.1 (4/22): Halting protocol implemented by user input.
-V1.5.0 (4/30): Removed UI for faster operation. Added energy balance computations. 
'''

# 0 IMPORT LIBRARIES
print("EXPLICIT FINITE DIFFERENCE HEAT TRANSFER SOLVER")
print("Written by Andrew Smith and Noah Etienne")
print("Version 1.5.0. Copyright 2023")
print('')
print('===================================================================')
print('')
import sys
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
import time



# 1 INPUT PARAMETERS
# 1.1 Properties of Aluminum
k = 237 # thermal conductivity [W/mK]
alpha = 97.1e-6 # thermal diffusivity [m^2/s]

# 1.2 Boundary Conditions
T0 = 20 # initial temperature of plate [deg C]
AT = 100 # temperature of edge A [deg C]
Bh = 45 # convective heat transfer coeff at edge B [W/m^2 K]
BT = 20 # bulk fluid temperature for convective, edge B [deg C]
CT = 300 # temperature of edge C [deg C]
Dq = 2e4 # heat flux at edge D [W/m^2]

# 1.3 Mesh Parameters
W = 30e-2 # length and height of plate [m]
H = 2e-2 # thickness of plate [m]
n_cells = 41 # number of cells in both X and Y direction
m = n_cells - 1 # index of last cell in X or Y direction
dx = W / n_cells # width of each cell

# 1.4 Time Settings
tmax = 500 # maximum time to compute heat transfer [s]
Fo = 0.1 # use maximum Fourier number for convergence
dt = Fo * dx**2 / alpha # define dt based on maximum Fo number (Fo < 0.25)
num_steps = int(tmax / dt) # total number of time steps

print("> SYS: Input parameters used: ")
print("  n_cells = " + str(n_cells) + ", t_max = " + str(tmax) )
#print(f"  frames per second: {(1/dt):.0f}")
num_comps = (m*(m-2)*num_steps) # total number of computations to perform
#print(f"  number of computations: {num_comps:.0f}")

const_a = 1.0402880017572e-6 # time in s per computation
const_b = 1.2836804192264e-1 # time constant for computation
exp_run_time = const_a*num_comps # approximated run time [s]. found using experimental values
if exp_run_time > 60: # show min and s
    print(f"  estimated runtime of simulation: {exp_run_time/60:.0f}" + "min" + f" {exp_run_time%60:.0f}" + "s")
elif exp_run_time < 1: # show ms
    print(f"  estimated runtime of simulation: {exp_run_time*1000:.0f}" + "ms")
elif exp_run_time < 5: # show s and ms
    print(f"  estimated runtime of simulation: {exp_run_time:.0f}" + "s" + f" {(1000*exp_run_time)%1000:.0f}" + "ms")
else: # just show s
    print(f"  estimated runtime of simulation: {exp_run_time:.0f}" + "s")



# 2 Helper Functions
Bi = Bh*dx/k # biot number

# 2.1 Temperature Conversions
C0_const = 273.15
def C_to_K(T_in_C):
    # convert scalar from celsius -> kelvin
    return (T_in_C + C0_const)
def K_to_C(T_in_K):
    # convert temp array from kelvin -> celsius
    return (np.array(T_in_K) - C0_const) 



# 3 MESH SURFACE
# define intial temperature to all cells
T = [[C_to_K(T0) for x in range(n_cells)] for y in range(n_cells)] # temperature array, 2D for both X and Y direction
'''
Note that array is indexed T[y][x] for position (x,y)
'''

# 3.1 Corner Temperatures
'''
Corner and select edge node temperatures are set manually
'''
T[0][0] = C_to_K(AT) # top left corner
T[0][m] = C_to_K(AT) # top right corner
T[m][0] = C_to_K(CT) # bottom left corner
T[m][m] = C_to_K(CT) # bottom right corner

# 3.2 Edge Temperatures
for i in range(1,n_cells-1): # skip first and last node
    T[0][i] = C_to_K(AT) # edge A
    T[m][i] = C_to_K(CT) # edge C
#print('> SYS: Initial temperatures:')
    
    
    
# 4 FINITE DIFFERENCE EQUATIONS
'''
Nomenclature:
Fo = fourier number
Bi = biot number
TC = current node temperature, T[y][x]
TL = left node temperature, T[y][x-1]
TR = right note temperature, T[y][x+1]
TU = upper node temperature, T[y-1][x]
TD = bottom node temperature, T[y+1][x]
'''
# 4.1 Interior Nodes
def int_T(Foo, TL, TR, TU, TB, TC):
    # computes T(p+1) [deg C] for Tp and Fo ("Foo") conditions
    return ( Foo*(TL+TR+TU+TB) + (1 - 4*Foo)*TC )

# 4.2 Edge Nodes
'''
Constant surface temperature conditions for edge A, C, and corners are specified later
'''
# 4.2.1 Edge B
def edge_B_T(Foo, TL, TU, TB, TC):
    return ( Foo*(2*TL + TU + TB + 2*Bi*BT) + (1 - 4*Foo - 2*Bi*Foo)*TC )

# 4.2.2 Edge D
def edge_D_T(Foo, TR, TU, TB, TC):
    return ( Foo*(TU + 2*TR + TB) + (1 - 4*Foo)*TC + (2*Foo*Bi*Dq)/Bh )

print('> SYS: Simulation setup completed.')




# 5 ITERATIVE CALCULATIONS
#temps = [T] # save temperature at each time step

print('> SYS: Performing simulation...')
start_time = time.time() #measure system wall time

for n in range(num_steps): # n current frame -- compute up to t_max
    #Fo = (alpha * dt) / (dx**2) # compute Fourier number
    
    # MAIN COMPUTATION
    for x in range(0, m+1): # loop over x direction (from 0 to m)
        for y in range(1, m): # loop over y direction (from 1 to m-1)
            # Compute Edge B and Edge D nodes
            if x == m: # edge B, corners are already ignored
                T[y][m] = Fo*(2*T[y][m-1] + T[y-1][m] + T[y+1][m] + 2*Bi*BT) + (1-4*Fo-2*Bi*Fo)*T[y][x]
            elif x == 0: # edge D, corners are already ignored
                T[y][x] = Fo*(T[y-1][0]+ 2*T[y][1] + T[y+1][0]) + (1-4*Fo)*T[y][0] + 2*Fo*Bi/Bh*Dq
            # Compute interior nodes
            else: # these must be interior nodes
                T[y][x] = Fo*(T[y][x-1]+T[y][x+1]+T[y-1][x]+T[y+1][x]) + (1-4*Fo)*T[y][x]
      
print('> SYS: Simulation Completed.')
end_time = time.time() # current system time
time_elapsed = end_time - start_time
if time_elapsed > 60: # show min and s
    print(f"  execution time: {exp_run_time/60:.0f}" + "min" + f" {exp_run_time%60:.0f}" + "s")
elif time_elapsed < 1: # show ms
    print(f"  execution time: {exp_run_time*1000:.0f}" + "ms")
elif time_elapsed < 5: # show s and ms
    print(f"  execution time: {exp_run_time:.0f}" + "s" + f" {(1000*exp_run_time)%1000:.0f}" + "ms")
else: # just show s
    print(f"  execution time: {exp_run_time:.0f}" + "s")    


'''
print('> SYS: Saving data to CSV...')
T = K_to_C(T) # convert to celsius
save_true = int(input("  Save temperature profile to CSV file? [1 = Yes, 0 = No]: "))
if save_true == 1:
    pd.DataFrame(T).to_csv('temps.csv')  # save file to csv
    print('> SYS: File saved as \"temps.csv\" in project directory.')
    print('  WARNING: Rename exported data as rerunning code will override data.') 
elif save_true == 0:
    print('> SYS: Saving overridden.')
'''



# 6 DISPLAY TEMPERATURE MAP
print('> SYS: Printing results.')
fig, ax1 = plt.subplots(1,1) # define figure and axis name
#img = plt.imshow(temps[num_steps], interpolation='bilinear') # plot temperature values
interpolation_type = 'bilinear' 
img = plt.imshow(T, interpolation = interpolation_type) # plot T, use to save memory (comment out line 132)

plot_title = 'n=' + str(n_cells) + ', t=' + str(tmax) + 's' # form title of graph
ax1.set_title(plot_title) # add graph title

img.axes.tick_params(color='black', labelcolor='black') # tickmarks
ax1.patch.set_facecolor('white') # face color
cb = plt.colorbar(img) # color bar

plt.tight_layout() # remove padding around figure
plt.show() # show plot



# 7 TEMPERATURE BALANCE
print('> SYS: Verifying simulation results.')

dA = dx*0.02 # surface area of each cell
qA = 0 # set as 0, add each cell
qB = 0
qC = 0 
# 7.1 CONSTANT TEMP EDGES
for i in range(len(T[0])): # loop over each column
    dTdx_A = (T[0][i] - T[1][i])/dx
    qA += k * dA * dTdx_A 
    
    dTdx_C = (T[m][i] - T[m-1][i])/dx
    qC += k * dA * dTdx_C
print(f"  qA =  {qA:.2f}" + "W")

# 7.2 CONVECTION EDGE
for i in range(1,m): # skip first and last row
    qB += Bh * dA * (BT - T[i][m])
print(f"  qB =  {qB:.2f}" + "W")
print(f"  qC =  {qC:.2f}" + "W")
qD = Dq * 0.30 * 0.02 # q = q'' * As
print(f"  qD =  {qD:.2f}" + "W")
print(f"  q TOTAL =  {(qA + qB + qC + qD):.2f}" + "W")

print('> SYS: Program completed.')

