# Reinforced learning - Agent for SnakeAI.py
import random
import snakeAI 
from snakeAI import SnakegameApp

# - Define some parameters for the ML machine

action = [] *  3    # [Straight, Right, Left]

state  = [] * 11    #  0 - Danger straight
                    #  1 - Danger right
                    #  2 - Danger left
                    #  3 - Direction left
                    #  4 - Direction right
                    #  5 - Direction up
                    #  6 - Direction down
                    #  7 - food left
                    #  8 - food right
                    #  9 - food up
                    # 10 - food down

# direction[x,y]    #   1,  0       Right
                    #  -1,  0       Left
                    #   0,  1       Down
                    #   0, -1       Up

# The agents hook into the timed loop in Tk for gaining control
def game_engine_hook():

    print("Here we are")

    if app.gamestatus != 'RUNNING':
        print(f'quitting:Status={app.gamestatus}')
        return

    # Fake user input for guidance
    next_dir = random.randint(0,4)
    if next_dir<3:
        action = [1,0,0]   # Straight
    elif next_dir == 3:
        action = [0,1,0]   # Right
    else:
        action = [0,0,1]   # Left

    # Get new direction from action
    x,y = app.direction

    if action[1] == 1:    # take right turn
        app.direction = [-y,  x]
    elif action[2] == 1:
        app.direction = [ y, -x]

    app.move()



# -- Start the game and enter the TK mainloop. 
app = SnakegameApp( game_engine_hook=game_engine_hook, 
                    running_ai=False)
app.run()

print("Game closed")
