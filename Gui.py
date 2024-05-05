import pygame, sys
from pygame.locals import *
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

# Constants
WINDOWSIZEX = 640
WINDOWSIZEY = 480
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Initialize Pygame
pygame.init()
pygame.font.init()
FONT = pygame.font.Font(None, 32)
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
pygame.display.set_caption('Digit Recognizer')

# Load and prepare model
model = load_model('digitRecognizer.h5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load dataset (for demonstration or testing purposes)
def load_dataset():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)) / 255.0
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)) / 255.0
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return X_train, X_test, y_train, y_test

def resize_image(image):
    image_surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
    image_surface = pygame.transform.smoothscale(image_surface, (28, 28))
    image_array = pygame.surfarray.array3d(image_surface)
    grayscale_image = np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140]) / 255.0
    return grayscale_image.reshape(1, 28, 28, 1)

# Event handling
is_writing = False
number_xcord = []
number_ycord = []

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == MOUSEMOTION and is_writing:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4)
            number_xcord.append(xcord)
            number_ycord.append(ycord)
        if event.type == MOUSEBUTTONDOWN:
            is_writing = True
        if event.type == MOUSEBUTTONUP:
            is_writing = False
            if number_xcord and number_ycord:
                min_x = max(min(number_xcord) - 10, 0)
                max_x = min(max(number_xcord) + 10, WINDOWSIZEX)
                min_y = max(min(number_ycord) - 10, 0)
                max_y = min(max(number_ycord) + 10, WINDOWSIZEY)
                number_xcord, number_ycord = [], []
                img_arr = pygame.surfarray.array3d(DISPLAYSURF)[min_x:max_x, min_y:max_y]
                img_arr = resize_image(img_arr)
                prediction = model.predict(img_arr)
                label = np.argmax(prediction)
                text_surface = FONT.render(str(label), True, RED, WHITE)
                text_rect = text_surface.get_rect(center=(min_x + (max_x - min_x) // 2, max_y + 20))
                DISPLAYSURF.blit(text_surface, text_rect)
        if event.type == KEYDOWN:
            if event.unicode == 'n':
                DISPLAYSURF.fill(BLACK)

    pygame.display.update()