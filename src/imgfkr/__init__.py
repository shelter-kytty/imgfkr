import imgfkr.core


def main() -> None:
    greyscale = cv2.imread("./data/in.jpg", cv2.IMREAD_GRAYSCALE)
    
    # contrast = createContrastMask(greyscaled, 40, 180)
    # edge = createEdgeMask(contrast)
    # moshed = sortPixels(edge, colourised, key=saturation, reverse=False)

    result = core.contrastMask(greyscale, 40, 180)

    cv2.imwrite("./data/out/out.jpg", result)