import pathlib
import pygubu
PROJECT_PATH = pathlib.Path(__file__).parent
PROJECT_UI = PROJECT_PATH / "snakegame.ui"

import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
import random, os
import numpy as np

from operator import itemgetter
import pickle

# - GAMEPLAN DATA SETUP
GAMEPLAN_SIZE     = 40     # Set the number of positions (boxes) that the snake can move (Horizontal and vertical)
GAMEBOX_SIZE      = 20     # Set the size, in pixels, for the width of the box
GAME_PADDING      = 10     # Set the padding on each side of the board
GAME_SPEED        = 150    # Set the default speed for the snake. Lower number is faster
SNAKE_LENGTH_ADD  = 3      # The increase in length each time the snake catch the cake     

class SnakegameApp:

    direction = [0,-1]
    position  = [0,0]
    candypos  = [0,0]
    gamedata = {
        'step'          : 0,
        'score'         : 0,
        'iteration'     : 0,
        'speed'         : GAME_SPEED,
        'delay'         : 0,
        'length'        : 5,
        'generation'    : 0,
    }

    snake_list = []
    gamematrix = np.zeros([GAMEPLAN_SIZE, GAMEPLAN_SIZE])

    def __init__(self, game_engine_hook, running_ai=False):

        self.gameplan_size    = GAMEPLAN_SIZE
        self.gamebox_size     = GAMEBOX_SIZE
        self.game_padding     = GAME_PADDING
        self.gamestatus       = 'FIRST TIME'
        self.running_ai       = running_ai
        self.game_engine_hook = game_engine_hook

        # -----------------------------------------------------------------------------------------
        # - PyGubu Builder Magic - Setup the static layout 
        # -----------------------------------------------------------------------------------------

        self.builder = builder = pygubu.Builder()
        builder.add_resource_path(PROJECT_PATH)
        builder.add_from_file(PROJECT_UI)
        # Main widget
        self.mainwindow = builder.get_object("toplevel1", None)
        self.canvas     = builder.get_object("canvas", None)
        self.info       = builder.get_object("info", None)
        self.highscore  = builder.get_object("highscore", None)
        
        self.score_variable = None
        self.info_variable = None
        self.highscore_variable = None
        self.scale_variable = None
        self.status_variable = None
        self.restart_variable = None
        builder.import_variables(self,
                                 ['score_variable',
                                  'info_variable',
                                  'highscore_variable',
                                  'scale_variable',
                                  'status_variable',
                                  'restart_variable'])

        builder.connect_callbacks(self)

        # Complete the gameplan in the canvas object --------------------------------------------

        for pos in range(0, self.gameplan_size+1, 1):
            xl = self.game_padding + pos*self.gamebox_size
            t  = self.gameplan_size*self.gamebox_size+self.game_padding

            self.canvas.create_line(xl, self.game_padding, xl, t)            
            self.canvas.create_line(self.game_padding, xl, t, xl)

        # - Get the snake head and body image from file (19x19), and cake image
        self.img_snake_body = ImageTk.PhotoImage(Image.open("snake_body.bmp"))
        self.img_snake_head = ImageTk.PhotoImage(Image.open("snake_head_up.bmp"))
        self.img_cake       = ImageTk.PhotoImage(Image.open("cake-19x19.bmp"))
        
        # - Bind some command buttons for shortcuts
        self.mainwindow.bind("<space>", self.spacebar_command)

        if running_ai == False: 
            # - Bind the arrow keys to the update move function
            self.mainwindow.bind("<Left>"  , lambda value : self.snake_update_move( -1,  0))
            self.mainwindow.bind("<Right>" , lambda value : self.snake_update_move( +1,  0))
            self.mainwindow.bind("<Up>"    , lambda value : self.snake_update_move(  0, -1))
            self.mainwindow.bind("<Down>"  , lambda value : self.snake_update_move(  0, +1))
        
    # - Run the mainloop 

    def run(self):
        self.mainwindow.mainloop()

    def start_new(self):
        
        # set the first position for the snake
        self.position[0] = random.randint(12, self.gameplan_size-12)   # '12' is just to set it not too close to edge
        self.position[1] = random.randint(12, self.gameplan_size-12)

        self.gamematrix[self.position[0], self.position[1]] = 1       # Upself.date the gamematrix, '1' means position is 'snake_head'

        self.snake = self.canvas.create_image( 
            self.position[0]*self.gamebox_size+self.game_padding+10,     # '10' is the image offset
            self.position[1]*self.gamebox_size+self.game_padding+10,
            image=self.img_snake_head,
            tags=('clean','snake'))

        # Set the first position for the cake, avoid putting it on-top of snake
        while True:
            self.candypos[0] = random.randint(5, self.gamebox_size-5)    # '5' is just to keep it away from edge
            self.candypos[1] = random.randint(5, self.gamebox_size-5)
            if not(self.candypos[0] == self.position[0] and self.candypos[1] == self.position[1]):
                break
        self.gamematrix[self.candypos[0], self.candypos[1]] = 3    # '3' indicates 'cake'

        self.candy = self.canvas.create_image(
            self.candypos[0]*self.gamebox_size+self.game_padding+10,
            self.candypos[1]*self.gamebox_size+self.game_padding+10,
            image=self.img_cake,
            tags=('clean', 'cake'))
        
    # --------------------------------------------------------------------------------------------
    # snake_update_move: depending on gamestatus, change the direction parameter
    # --------------------------------------------------------------------------------------------
    def snake_update_move(self, x, y):
        if (self.direction[0] != -x) and (self.direction[1] != -y):
            self.direction[0], self.direction[1] = x, y

    def check_collision(self, x, y): 
        # Check wall crash
        if ((x < 0 or x>self.gameplan_size-1) or (y < 0 or y>self.gameplan_size-1)):
            return 'CRASH_WALL'

        # Check if we crashed with snake body
        if self.gamematrix[x,y] == 2:
            return 'CRASH_SNAKE'

        # Check if we have done to many iterations since last cake was found
        if (len(self.snake_list) > 2) and (self.gamedata['iteration'] > 100*len(self.snake_list)):
            return 'CRASH_ITERATION'
            
        return 'NO_COLLISION'

    def check_candy_found(self, x, y):
        if self.candypos == [x,y]:
            return True
        
        return False
    
    # --------------------------------------------------------------------------------------------
    # move: Move the snake, and update status information
    # --------------------------------------------------------------------------------------------
    def move(self):
        
        # Save the old position, and update position with the new direction
        prevpos = [0,0]
        prevpos[0] = self.position[0]  
        prevpos[1] = self.position[1]
        
        self.position[0] += self.direction[0]
        self.position[1] += self.direction[1]

        # Check if we hit the wall or the body
        is_collision = self.check_collision(self.position[0], self.position[1])

        if is_collision != 'NO_COLLISION':
            self.gamestatus = is_collision            
            
            if is_collision == 'CRASH_WALL' :
                self.info_variable.set("GAME OVER!\nYou hit the wall!\nPress 'Start Game' to go again.")
            elif is_collision == 'CRASH_SNAKE':
                self.info_variable.set("GAME OVER!\nYou ran into yourself!\nPress 'Start Game' to go again.")
            else:
                self.info_variable.set("GAME OVER!\nYou are stuck!\nPress 'Start Game' to go again.")
            self.info.config(bg='red')
            return

        # Check if we caught the candy 
        if self.check_candy_found(self.position[0], self.position[1]) == True:
            self.gamedata['score'] += 1
            self.score_variable.set(f"{self.gamedata['score']:03d}")
            self.gamestatus = 'CAKE_FOUND'
            self.gamedata['iteration'] = 0
            # - place the cake again, be sure that is not set in an occupied box
            while True:
                self.candypos[0] = random.randint(5, self.gameplan_size-5)
                self.candypos[1] = random.randint(5, self.gameplan_size-5)
                if self.gamematrix[self.candypos[0], self.candypos[1]] == 0:
                    break
                else:  # For debugging, TODO
                    print("CANNOT FIND CAKE POSITION")

            self.gamematrix[self.candypos[0], self.candypos[1]] = 3
            self.canvas.moveto(
                self.candy, 
                self.candypos[0]*self.gamebox_size+self.game_padding+1,
                self.candypos[1]*self.gamebox_size+self.game_padding+1
            )
            self.gamedata['length'] += SNAKE_LENGTH_ADD

        # Update new position in gamematrix
        self.gamematrix[prevpos[0], prevpos[1]] = 0    # Clear old position. (Actually not needed)
        self.gamematrix[self.position[0], self.position[1]] = 1

        # Move the image_head of the snake to it's new position
        self.canvas.move(self.snake,
            self.direction[0]*self.gamebox_size,
            self.direction[1]*self.gamebox_size
        )   
        
        # Move each image section of the snake body to it's new positions
        if len(self.snake_list) < self.gamedata['length']:    # Need to create new body segments
            body = self.canvas.create_image(
                prevpos[0]*self.gamebox_size + self.game_padding + 10,
                prevpos[1]*self.gamebox_size + self.game_padding + 10,
                image=self.img_snake_body,
                tags=('clean', 'body')
            )
            self.snake_list.insert(0,[body, prevpos[0], prevpos[1]])
            self.gamematrix[prevpos[0], prevpos[1]] = 2
        else:
            item = self.snake_list.pop()
            self.gamematrix[item[1], item[2]]=0    # Clear old positon in matrix
            item[1], item[2] = prevpos[0], prevpos[1]
            self.snake_list.insert(0, item)
            self.canvas.moveto( item[0],
                                item[1]*self.gamebox_size+self.game_padding,
                                item[2]*self.gamebox_size+self.game_padding)
            self.gamematrix[prevpos[0], prevpos[1]]=2

        # Update the step counter (Need to present it in the dashboard later ...)
        self.gamedata['step'] += 1
        self.gamedata['iteration'] += 1

        self.status_variable.set(
            f"Status:\n\nstep: {self.gamedata['step']}\n" +\
            f"iteration: {self.gamedata['iteration']}\n" +\
            f"Generation: {self.gamedata['generation']}\n")

    
    # --------------------------------------------------------------------------------------------

    def game_engine(self):
        if self.gamestatus == 'GAMEOVER':
            return
        
        self.game_engine_hook()

        # self.move()

        
        self.mainwindow.after(int(self.gamedata['delay']),self.game_engine)
        # self.mainwindow.after(300,self.game_engine)
    # ---------------------------------------------------------------------------------------------

    def spacebar_command(self, event):
        self.restart_command()

    def setspeed_command(self, scale_value):
        pass

    def restart_command(self, dont_restart_engine=False):
        
        # -- Clean up previous session
        # ----- Delete all objects
        self.canvas.delete('clean')
        self.gamematrix.fill(0)
        self.gamedata['step']   = 0
        self.gamedata['score']  = 0
        self.gamedata['speed']  = self.scale_variable.get() 
        self.gamedata['delay']  = (490-40*self.scale_variable.get())/3
        
        #print(f"scale: {self.scale_variable.get()} speed: {self.gamedata['speed']}")

        self.gamedata['length'] = 5
        self.snake_list = []

        self.info_variable.set("\n      Catch the Cake ....")
        self.info.config(bg="green")

        self.score_variable.set(f"{self.gamedata['score']:03d}")

        # -- Start up   game session 
        self.gamestatus = 'RUNNING'
        self.start_new()
        if dont_restart_engine == False:
            self.game_engine()
        
if __name__ == "__main__":
    app = SnakegameApp(GAMEPLAN_SIZE, GAMEBOX_SIZE, GAME_PADDING)
    app.run()

# %%
