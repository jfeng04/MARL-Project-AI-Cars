This project is a combined project that I made with Keg Avakian in the UW-Madison in a project under the supervision of Young Wu. It is to be noted that core aspects of this project are done in collaboration
with Keg, however any additions/changes after the initial commit in this repository are 100 percent my own.

To use this program is simple: in the MARL-cars main directory, look for the simulator.html file, this is where you right-click the file and press 'Open with Live Server' with. If you do not have Live Server,
please download it on VSCode. 

Description of AI: Cars are connected each to a neural network and are ultimately connected to a major global neural network, each car fulfilling a specific node and having their own weights. They gain 'rewards'
by the amount of distance they travelled and the speed they are going (which ultimately ties into the amount of distance they've travelled), crashing or bouncing off of other cars will result in a negative reward,
and distance travelled is key to their positive reward to prevent a car from doing circles around itself. 

Any further contributions must be on a seperate branch from my main if you are willing to push onto this repository. If you have any questions, contact me at: jfeng04@outlook.com

If you are simply a user, please refer to this for controls:

- Slider: Change the speed of time
- Pause: Pause the program
