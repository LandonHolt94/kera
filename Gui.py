import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model

WINDOWSIZEX = 640
WINDOWSIZEY = 480

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

MODEL = load_model('digitRecognizer.h5')

LABELS = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}

pygame.init()
pygame.font.init()
FONT = pygame.font.Font(None, 32)
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
pygame.display.set_caption('Digit Recognizer')

def resize_image(image):
    """Resize and normalize the image to fit the model input."""
    image_surface = pygame.surfarray.make_surface(image)
    image_surface = pygame.transform.smoothscale(image_surface, (28, 28))  # Resize image to 28x28
    image_array = pygame.surfarray.array3d(image_surface)
    grayscale_image = np.mean(image_array, axis=2) / 255.0  # Convert to grayscale and normalize
    return grayscale_image

iswriting = False
number_xcord = []
number_ycord = []

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)
            number_xcord.append(xcord)
            number_ycord.append(ycord)
        if event.type == MOUSEBUTTONDOWN:
            iswriting = True
        if event.type == MOUSEBUTTONUP:
            iswriting = False
            min_x = max(min(number_xcord) - 10, 0)
            max_x = min(max(number_xcord) + 10, WINDOWSIZEX)
            min_y = max(min(number_ycord) - 10, 0)
            max_y = min(max(number_ycord) + 10, WINDOWSIZEY)
            number_xcord, number_ycord = [], []

            img_arr = pygame.surfarray.array3d(DISPLAYSURF)[min_x:max_x, min_y:max_y].transpose([1, 0, 2])
            img_arr = resize_image(img_arr)

            label = LABELS[np.argmax(MODEL.predict(img_arr.reshape(1, 28, 28, 1)))]
            text_surface = FONT.render(label, True, RED, WHITE)
            text_rect = text_surface.get_rect()
            text_rect.left, text_rect.bottom = min_x, max_y
            DISPLAYSURF.blit(text_surface, text_rect)
        if event.type == KEYDOWN:
            if event.unicode == 'n':
                DISPLAYSURF.fill(BLACK)

    pygame.display.update()