## Convolution laboratory: Signals and Systems - Systems Modeling

Python program in which the convolution process is developed in both the discrete time domain and the continuous time domain. 

#### The requirements are as follows: 

The signals that may be chosen as input (x) and/or response to impulse (h) for the convolution shall be:
1. Quadratic signal
2. Sinusoidal signal 
3. Triangular signal 
4. Base log signal 10
6. Ramp sign
7. Token to pieces of free choice (Define it in interface and video).


- These signals can be chosen from a drop-down menu. The user is shown the definition (formula of each signal).
- The signals representing input and impulse response are shown graphically. 
- The user can enter the parameters of the functions to be convoluted (amplitude, start point ti and end point tf of each function). 
- For the quadratic function, the values A, B and C of its definition can be modified. 
- For sinusoidal functions, no matter the starting point, a complete cycle is always displayed, for this, the parameter that will have to be modified is the period. 
- In the case of the Ramp sign, it consists of three sections which are divided proportionally. This proportion should not be altered when the user changes the time interval. These signals do not require a change in amplitude. 
- In the case of discrete convolution, when representing the sinusoidal function, it is constructed with sufficient samples. Similarly a complete cycle of it is shown. (In this case the signals are plotted using the stem function). 
- All functions can be used together for convolution.
