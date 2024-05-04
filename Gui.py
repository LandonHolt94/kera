import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model

WINDOWSIZEX = 640
WINDOWSIZEY = 480

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

IMAGESAVE = False

MODEL = load_model('digitRecognizer.h5')

LABEL = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9'}

pygame.init()


FONT = pygame.font.FontTyper('freesansbold.ttf', 24)
pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))

pygame.display.set_caption('Digit Recognizer')


while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()