import numpy as np

def get_limits(color):
    # If the color is red, handle both low and high ranges, red was having trouble, this was the fix
    if isinstance(color[0], list):
        lowerLimit1 = np.array([max(color[0][0] - 10, 0), 100, 100], dtype=np.uint8)
        upperLimit1 = np.array([min(color[0][0] + 10, 179), 255, 255], dtype=np.uint8)

        lowerLimit2 = np.array([max(color[1][0] - 10, 0), 100, 100], dtype=np.uint8)
        upperLimit2 = np.array([min(color[1][0] + 10, 179), 255, 255], dtype=np.uint8)

        return (lowerLimit1, upperLimit1), (lowerLimit2, upperLimit2)

    # Extract hue value
    hue = color[0]

    # Check if hue corresponds to blue, yellow, or purple. Blue, yellow, and purple werent registering well, while the rest were so i increased the hue range
    if hue in [120, 35, 150]:  # Blue = 120, Yellow = 35, Purple = 150
        lowerLimit = np.array([max(hue - 25, 0), 100, 100], dtype=np.uint8)
        upperLimit = np.array([min(hue + 25, 179), 255, 255], dtype=np.uint8)
    else:
        lowerLimit = np.array([max(hue - 10, 0), 100, 100], dtype=np.uint8)
        upperLimit = np.array([min(hue + 10, 179), 255, 255], dtype=np.uint8)

    return lowerLimit, upperLimit