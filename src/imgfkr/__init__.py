import imgfkr.core
import cv2
import numpy as np


def main() -> None:
    greyscale = cv2.imread("./data/in.jpg", cv2.IMREAD_GRAYSCALE)
    
    result = core.contrastMask(greyscale, 40, 180)

    cv2.imwrite("./data/out/out.jpg", result)