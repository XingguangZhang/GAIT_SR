import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

import sys
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

verticies_demo = (
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, -1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, -1, 1),
    (-1, 1, 1)
)

edges_demo = (
    (0, 1),
    (0, 3),
    (0, 4),
    (2, 1),
    (2, 3),
    (2, 7),
    (6, 3),
    (6, 4),
    (6, 7),
    (5, 1),
    (5, 4),
    (5, 7)
)

Body_25 = (
    (0, 1), (1, 2), (2, 3), (3, 4), (1, 5),
    (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
    (10, 11),  # (11, 22),  # (22,23), (11,24),
    (8, 12),
    (12, 13), (13, 14),  # (14,19), (19,20), (14,21),
    (0, 15),  # (15,17),
    (0, 16),  # (16,18)
)


def Cube(verticies, edges):
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(verticies[vertex])
    glEnd()


def DrawSkeleton(verticies, edges):
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(verticies[vertex])
    glEnd()


"""
def TransWorld(pos):
    NewPos = []
    if isinstance(pos, list):
        for i in range(len(pos)):
            (x, y, z) = pos[i]
            z = z / 256 * 4.5
            r = z / 4.5
            TotalHeight = 3.2 * r
            TotalWidth = 3.8 * r
            NewX = TotalHeight / 424 * x - TotalHeight
            NewY = TotalWidth / 512 * y - TotalWidth
            NewPos.append((NewX, -NewY, (r - 3) / 4.5))
    return NewPos
"""


def TransWorld(pos):
    NewPos = []
    if isinstance(pos, list):
        for i in range(len(pos)):
            (x, y, z) = pos[i]
            r = z / 128
            x = x * r
            y = y * r
            NewX = (x - 212) / 212
            NewY = (y - 256) / 256
            NewZ = z / 128
            NewPos.append((NewX, -NewY, NewZ))
    # print('received pos:', pos)
    print('Display:', NewPos)
    return NewPos


def GenLines(verticies, edges):
    i = 0
    for edge in edges:
        (x_start, y_start, z_start) = verticies[edge[0]]
        (x_end, y_end, z_end) = verticies[edge[1]]
        if i == 0:
            x_cor = np.linspace(x_start, x_end, 100)
            y_cor = np.linspace(y_start, y_end, 100)
            z_cor = np.linspace(z_start, z_end, 100)
        else:
            x_cor = np.concatenate((x_cor, np.linspace(x_start, x_end, 100)))
            y_cor = np.concatenate((y_cor, np.linspace(y_start, y_end, 100)))
            z_cor = np.concatenate((z_cor, np.linspace(z_start, z_end, 100)))
        i += 1
    return x_cor, y_cor, z_cor


def ShowMat(BreakKinect, skeleton_recv, verticies=verticies_demo, edges=edges_demo):
    def data_gen():
        while True:
            sk = skeleton_recv.recv()
            if isinstance(sk, list) and len(sk) == 25:
                x_cor, y_cor, z_cor = GenLines(sk, Body_25)
                yield (x_cor, y_cor, z_cor)

            fig = plt.figure()
            ax = Axes3D(fig)

            if isinstance(sk, list) and len(sk) == 25:
                x_cor, y_cor, z_cor = GenLines(sk, Body_25)
                print("x_cor shape:", x_cor.shape)
                ax.plot(x_cor, y_cor, z_cor)
                plt.show()

    if BreakKinect.value:
        sys.exit()


def ShowSkeleton(skeleton_recv, verticies=verticies_demo, edges=edges_demo):
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    gluPerspective(60, (display[0] / display[1]), 0.1, 10.0)

    glTranslatef(0.0, 0.0, -5)
    glEnable(GL_DEPTH_TEST)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        sk = skeleton_recv.recv()
        if isinstance(sk, list) and len(sk) == 25:
            sk = TransWorld(sk)
            # glRotatef(1, 3, 1, 1)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            DrawSkeleton(sk, Body_25)
            pygame.display.flip()
            pygame.time.wait(10)
        if sk == "Break":
            print("pygame is quitting...")
            break
    pygame.quit()


'''
        else:
            glRotatef(1, 3, 1, 1)
            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
            Cube(verticies, edges)
            pygame.display.flip()
            pygame.time.wait(10)
'''
