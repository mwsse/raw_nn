# Reinforced learning - Agent for SnakeAI.py
import threading
import snakeAI 
from snakeAI import SnakegameApp

# - GAMEPLAN DATA SETUP
GAMEPLAN_SIZE     = 40     # Set the number of positions (boxes) that the snake can move (Horizontal and vertical)
GAMEBOX_SIZE      = 20     # Set the size, in pixels, for the width of the box
GAME_PADDING      = 10     # Set the padding on each side of the board
GAME_SPEED        = 150    # Set the default speed for the snake. Slower number is faster
SNAKE_LENGTH_ADD  = 3      # The increase in length each time the snake catch the cake     

#   bind spacebar_command, 
#   def spacebar_command(self, event):
#        self.restart_command()

def spacebar_cmd(event):
    global app
    print('Here I am :) ')
    app.restart_command()

def game_engine_hook():

    print("..and again ... ")

    if app.gamestatus != 'RUNNING':
        print(f'quitting:Status={app.gamestatus}')
        start_time.cancel
        #if app.gamestatus== "WINDOW KILL":                
            # app.quit()
            # start_time.quit
                
        return
    
    app.move()

    # start_time = threading.Timer(150.0/1000, game_engine)
    # start_time.start()

def at_close():
    global start_time

    print('quitting ...')
    # app.destroy
    # start_time.quit

app = SnakegameApp(GAMEPLAN_SIZE, GAMEBOX_SIZE, GAME_PADDING, game_engine_hook=game_engine_hook)

app.run()


print("When will this be executed?")
#start_time
# app.at_close()