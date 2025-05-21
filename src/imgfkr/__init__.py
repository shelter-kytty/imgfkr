import imgfkr.core
import cv2
import numpy as np


def main() -> None:
    greyscale = cv2.imread("./data/in.jpg", cv2.IMREAD_GRAYSCALE)
    colour = cv2.imread("./data/in.jpg", cv2.IMREAD_COLOR)
    
    result = core.contrastMask(greyscale, 20, 180)

    cv2.imwrite("./data/out/out2.jpg", result)

    result = core.hSort(colour, result)

    cv2.imwrite("./data/out/out1.jpg", result)