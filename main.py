import cv2
from matplotlib import pyplot
import numpy as np
import random

def main():
    pyplot.figure(figsize=(24, 11))
    samoloty = ['samoloty/samolot00.jpg', 'samoloty/samolot01.jpg', 'samoloty/samolot02.jpg',
                'samoloty/samolot03.jpg', 'samoloty/samolot04.jpg', 'samoloty/samolot05.jpg',
                'samoloty/samolot06.jpg', 'samoloty/samolot07.jpg', 'samoloty/samolot08.jpg',
                'samoloty/samolot09.jpg', 'samoloty/samolot10.jpg', 'samoloty/samolot11.jpg',
                'samoloty/samolot12.jpg', 'samoloty/samolot13.jpg', 'samoloty/samolot14.jpg',
                'samoloty/samolot15.jpg', 'samoloty/samolot16.jpg', 'samoloty/samolot17.jpg']

    for i, samolot in enumerate(samoloty):
        i = i + 1
        image = cv2.imread(samolot)
        newImage = image.copy()
        # (img/średnica sąsiedztwa pixkseli/
        # /Im większa wartość, tym kolory znajdujące się dalej od siebie zaczną się mieszać/
        # /Im większa jego wartość, tym więcej pikseli będzie się ze sobą mieszać)
        bilateral_image = cv2.bilateralFilter(newImage, 40, 10, 10)

        # Skala szarości
        gray = cv2.cvtColor(bilateral_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Poszukiwanie krawędzi Canny
        edged = cv2.Canny(blurred, 50, 200,L2gradient = True)
        # Poszukiwanie konturów
        # edged-obraz źródłowy, cv2.RETR_EXTERNAL-tryb pobierania konturów, cv2.CHAIN_APPROX_SIMPLE-metoda aproksymacji konturów.
        contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for j in range(0, len(contours)):
            end_picture = cv2.drawContours(newImage, contours, j, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 3)

        plot = pyplot.subplot(6, 3, i)
        pyplot.axis('off')
        pyplot.tight_layout(pad=0)
        pyplot.subplots_adjust(wspace=None, hspace=None)
        plot.imshow(end_picture, cmap=pyplot.cm.gray)

    pyplot.savefig('zad-1.jpg', bbox_inches='tight', pad_inches=0)

if __name__ == '__main__':
    main()
