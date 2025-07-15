# MK_AI
A tool to train Mario Kart Double Dash with Dolphin emulator

# HOW TO SET IT UP
- Open Dolphin Emulator
- Ensure you have VJoy set up. [A detailed explanation on how to install VJoy can be found here](https://youtu.be/Mf8e4GnZCS4).
  You will need to install pyvjoy and run commands to configure it for use with dolhpin. [Here is another tutorial](https://youtu.be/W6CFPW4go0M)
- You will need pymem, mss, numpy, pickle and pyvjoy installed. pip install them if necessary
- With the controller set up and Dolphin running Double Dash, start the program. It should soft reset and immediately start.
  if you do not go straight to time trial on Luigi Circuit with Mario and Luigi selected, restart the program.

### HOW IT WORKS
This model uses Q-Table reinforcment learning to train the game. It connects to Dolphin's virtual memory and gets many values, including:
- Speed
- Acceleration
- Lap Progress
- Lap Time
- Terrain
- Dacing Direction
- Coordinates

All together, it uses those metrics to learn based on a reward system. This reward system says the following:
- Going in the right direction will give a positive reward while the wrong direction gives a negative reward
- The faster you go while heading in the positive direction, the more reward you recieve
- Recieve a negative reward if you are not on the racetrack (offroading in other words)

The state and reward system gets logged 10 times per second. Every 4000 updates, it soft resets to ensure it does not get stuck.

### YOUTUBE VIDEO ON THIS COMING SOON!
