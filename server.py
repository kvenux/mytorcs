#!/usr/bin/env python

import socket
import pygame
import math
import time

#initialize game engine
pygame.init()

#Open a window
size = (800, 500)
screen = pygame.display.set_mode(size)

#Set title to the window
pygame.display.set_caption("Hello World")

dead=False

#Initialize values for color (RGB format)
WHITE=(255,255,255)
RED=(255,0,0)
GREEN=(0, 255, 0)
BLUE=(0,0,255)
BLACK=(0,0,0)

# clock = pygame.time.Clock()
PI=math.pi
step = 0

TCP_IP = '127.0.0.1'
TCP_PORT = 5005
BUFFER_SIZE = 1024  # Normally 1024, but we want fast response

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(1)

conn, addr = s.accept()
print('Connection address:', addr)
while 1:
    data = conn.recv(BUFFER_SIZE)
    # print(data)
    try:
        arr = data.split(',')
        arr = map(float, arr)
        screen.fill(WHITE)
        width = 5

        for i in range(19):
            start_angle = -5.0/180.0*PI + i*10.0/180.0*PI
            end_angle = start_angle + 10.0/180.0*PI
            alpha = PI - start_angle
            beta = PI - end_angle
            radius = 200 + int(600*arr[i])
            # radius = 300 + int(20*i)
            # print(radius)
            if(radius < 10):
                radius = 20
            if i < 6:
                color = RED
            elif i < 12:
                color = GREEN
            else:
                color = BLUE
            pygame.draw.arc(screen, color, [400 - radius/2, 400 - radius/2, radius, radius], beta, alpha, width)

        pygame.draw.line(screen, RED, (400, 250), (400 + 100*math.sin(arr[19]), 250+ 100*math.cos(arr[19])), 5)
        pygame.display.update()
        time.sleep(0.1)
    except:
        pass

conn.close()