# Purpose
The purpose of this repository is render images or animations as stereo sound. This stereo sound can be then visualized in an oscilloscope. Search for "oscilloscope music"
on the internet for reference.

# Commands
The script initializes to the following state:
> f:mesh status:on name:octahedron
> f:mic status:on
> f:waves status:on n:0 wf:50 rate:0.001
> f:rotx status:on n: 0, angle: 0, rate:0.0030
> f:roty status:on n: 0, angle: 0, rate:0.0020
> f:rotz status:on n: 0, angle: 0, rate:0.0010

The user can change the state of the script by entering the following command in the terminal -while the script is running-:

> function parameter value

Here are a few command examples:
1. > mesh name cube
2. > waves status off
3. > rotx rate 0.01

Notes:
1. Microphone function only works when the script is executed from the terminal. This has to do with needing admin permissions.
2. Only one parameter can be changed at a time.
