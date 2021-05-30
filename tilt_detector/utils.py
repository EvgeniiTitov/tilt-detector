import os

import cv2
import numpy as np


def calculate_angle(the_lines: list) -> float:
    """
    Calculates angle of the line(s) provided
    :param the_lines: list of lists, lines found and filtered
    :return: angle
    """
    if len(the_lines) == 2:
        x1_1 = the_lines[0][0][0]
        y1_1 = the_lines[0][0][1]
        x2_1 = the_lines[0][1][0]
        y2_1 = the_lines[0][1][1]

        # Original approach
        # angle_1 = round(90 - np.rad2deg(np.arctan2(abs(y2_1 - y1_1),
        #                                 abs(x2_1 - x1_1))), 2)

        angle_1 = round(
            np.rad2deg(np.arctan(abs(x2_1 - x1_1) / abs(y2_1 - y1_1))), 2
        )

        x1_2 = the_lines[1][0][0]
        y1_2 = the_lines[1][0][1]
        x2_2 = the_lines[1][1][0]
        y2_2 = the_lines[1][1][1]

        # Original approach
        # angle_2 = round(90 - np.rad2deg(np.arctan2(abs(y2_2 - y1_2),
        #                                            abs(x2_2 - x1_2))), 2)

        angle_2 = round(
            np.rad2deg(np.arctan(abs(x2_2 - x1_2) / abs(y2_2 - y1_2))), 2
        )

        return round((angle_1 + angle_2) / 2, 2)

    else:
        x1 = the_lines[0][0][0]
        y1 = the_lines[0][0][1]
        x2 = the_lines[0][1][0]
        y2 = the_lines[0][1][1]

        # Original approach
        # return round(90 - np.rad2deg(np.arctan2(abs(y2 - y1),
        #                                         abs(x2 - x1))), 2)
        return round(np.rad2deg(np.arctan(abs(x2 - x1) / abs(y2 - y1))), 2)


def draw_lines_write_text(lines, image, angle):
    for line in lines:
        cv2.line(
            image,
            (line[0][0], line[0][1]),
            (line[1][0], line[1][1]),
            (0, 0, 255),
            2,
        )
    cv2.putText(
        image,
        str(angle),
        (int(image.shape[1] * 0.35), int(image.shape[0] * 0.95)),
        fontScale=1,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        color=(255, 0, 255),
        lineType=3,
        thickness=2,
    )

    return image


def save_image(save_path, lines, image, image_name, angle):

    cv2.imwrite(
        os.path.join(save_path, image_name),
        draw_lines_write_text(lines, image, angle),
    )


def save_image_2(save_path, image_name, image):
    cv2.imwrite(os.path.join(save_path, image_name), image)


def show_image(image):
    cv2.imshow("Extracted Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
