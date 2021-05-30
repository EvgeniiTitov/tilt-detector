import math

import cv2
import imutils
import numpy as np
from utils import calculate_angle
from utils import draw_lines_write_text


class PoleAngleInclination:
    def __init__(self):
        self._angle_thresh = 70
        self._min_distance_to_merge = 30
        self._min_angle_to_merge = 30

    def extend_lines(self, lines_to_extend: list, image: np.array) -> list:
        """
        :param lines_to_extend:
        :return:
        """
        lines_extended = list()

        if len(lines_to_extend) == 1:
            for line in lines_to_extend:
                x1, y1 = line[0][0], line[0][1]
                x2, y2 = line[1][0], line[1][1]

                curr_lenght = math.sqrt((x1 - x2) ** 2 + (y2 - y1) ** 2)

                # y = 0
                x_top = int(round(x1 + (x1 - x2) / curr_lenght * y1))

                # y = image.shape[0]
                x_bottom = int(
                    round(x2 + (x2 - x1) / curr_lenght * (image.shape[0] - y2))
                )
                # Dots are intentionally appended *flat* to the list,
                # not typical syntax (x1,y1), (x2,y2) etc
                # lines_extended.append([(x_top, 0), (x_bottom, img.shape[0])])
                lines_extended.append([x_top, 0])
                lines_extended.append([x_bottom, image.shape[0]])

        else:
            # If the algorithm failed to find two lines and returned only one,
            # we draw approximate second line to extract the area in between
            # First extend the line detected by the algorithm
            x1, y1 = lines_to_extend[0][0][0], lines_to_extend[0][0][1]
            x2, y2 = lines_to_extend[0][1][0], lines_to_extend[0][1][1]

            curr_lenght = math.sqrt((x1 - x2) ** 2 + (y2 - y1) ** 2)

            # y = 0
            x_top = int(round(x1 + (x1 - x2) / curr_lenght * y1))

            # y = image.shape[0]
            x_bottom = int(
                round(x2 + (x2 - x1) / curr_lenght * (image.shape[0] - y2))
            )

            lines_extended.append([x_top, 0])
            lines_extended.append([x_bottom, image.shape[0]])

            # Draw second approximate line parallel to the first one.
            x_new_top = image.shape[1] - x_bottom
            x_new_bottom = image.shape[1] - x_top

            lines_extended.append([x_new_top, 0])
            lines_extended.append([x_new_bottom, image.shape[0]])

        return lines_extended

    def merge_lines(self, lines_to_merge):
        """
        Merges lines provided they are similarly oriented
        :param lines_to_merge: List of lists. Lines to merge
        :return: merged lines
        """
        vertical_lines = self.discard_not_vertical_lines(lines_to_merge)

        # Sort and get line orientation
        lines_x, lines_y = list(), list()

        for line in vertical_lines:
            orientation = math.atan2(
                (line[0][1] - line[1][1]), (line[0][0] - line[1][0])
            )

            if (abs(math.degrees(orientation)) > 45) and abs(
                math.degrees(orientation)
            ) < (90 + 45):
                lines_y.append(line)
            else:
                lines_x.append(line)

        lines_x.sort(key=lambda line: line[0][0])
        lines_y.sort(key=lambda line: line[0][1])

        merged_lines_x = self.merge_lines_pipeline_2(lines_x)
        merged_lines_y = self.merge_lines_pipeline_2(lines_y)

        merged_lines_all = list()
        merged_lines_all.extend(merged_lines_x)
        merged_lines_all.extend(merged_lines_y)

        return merged_lines_all

    def discard_not_vertical_lines(self, lines):
        """
        Discards all lines that are not within N degrees cone
        :param lines:
        :return: vertical lines [(point1, point2, angle of the line), ]
        """
        # Discard horizontal lines
        vertical_lines = list()

        for line in lines:

            x1 = line[0][0]
            y1 = line[0][1]
            x2 = line[0][2]
            y2 = line[0][3]

            angle = abs(round(np.rad2deg(np.arctan2((y2 - y1), (x2 - x1))), 2))

            if angle < self._angle_thresh:
                continue

            vertical_lines.append([(x1, y1), (x2, y2)])

        return vertical_lines

    def merge_lines_pipeline_2(self, lines):
        """

        :param lines:
        :return:
        """
        super_lines_final = []
        super_lines = []

        # check if a line has angle and enough distance
        # to be considered similar
        for line in lines:
            create_new_group = True
            group_updated = False

            for group in super_lines:
                for line2 in group:

                    if (
                        self.get_distance(line2, line)
                        < self._min_distance_to_merge
                    ):
                        # check the angle between lines
                        orientation_i = math.atan2(
                            (line[0][1] - line[1][1]),
                            (line[0][0] - line[1][0]),
                        )
                        orientation_j = math.atan2(
                            (line2[0][1] - line2[1][1]),
                            (line2[0][0] - line2[1][0]),
                        )

                        if (
                            int(
                                abs(
                                    abs(math.degrees(orientation_i))
                                    - abs(math.degrees(orientation_j))
                                )
                            )
                            < self._min_angle_to_merge
                        ):
                            # print("angles", orientation_i, orientation_j)
                            # print(int(abs(orientation_i - orientation_j)))
                            group.append(line)

                            create_new_group = False
                            group_updated = True
                            break

                if group_updated:
                    break

            if create_new_group:
                new_group = list()
                new_group.append(line)

                for idx, line2 in enumerate(lines):
                    # check the distance between lines
                    if (
                        self.get_distance(line2, line)
                        < self._min_distance_to_merge
                    ):

                        # check the angle between lines
                        orientation_i = math.atan2(
                            (line[0][1] - line[1][1]),
                            (line[0][0] - line[1][0]),
                        )
                        orientation_j = math.atan2(
                            (line2[0][1] - line2[1][1]),
                            (line2[0][0] - line2[1][0]),
                        )

                        if (
                            int(
                                abs(
                                    abs(math.degrees(orientation_i))
                                    - abs(math.degrees(orientation_j))
                                )
                            )
                            < self._min_angle_to_merge
                        ):
                            # print("angles", orientation_i, orientation_j)
                            # print(int(abs(orientation_i - orientation_j)))

                            new_group.append(line2)

                            # remove line from lines list
                            # lines[idx] = False
                # append new group
                super_lines.append(new_group)

        for group in super_lines:
            super_lines_final.append(self.merge_lines_segments1(group))

        return super_lines_final

    def merge_lines_segments1(self, lines, use_log=False):

        if len(lines) == 1:
            return lines[0]

        line_i = lines[0]

        # orientation
        orientation_i = math.atan2(
            (line_i[0][1] - line_i[1][1]), (line_i[0][0] - line_i[1][0])
        )

        points = []
        for line in lines:
            points.append(line[0])
            points.append(line[1])

        if (abs(math.degrees(orientation_i)) > 45) and abs(
            math.degrees(orientation_i)
        ) < (90 + 45):

            # sort by y
            points = sorted(points, key=lambda point: point[1])

            if use_log:
                print("use y")
        else:

            # sort by x
            points = sorted(points, key=lambda point: point[0])

            if use_log:
                print("use x")

        return [points[0], points[len(points) - 1]]

    def lines_close(self, line1, line2):

        dist1 = math.hypot(
            line1[0][0] - line2[0][0], line1[0][0] - line2[0][1]
        )
        dist2 = math.hypot(
            line1[0][2] - line2[0][0], line1[0][3] - line2[0][1]
        )
        dist3 = math.hypot(
            line1[0][0] - line2[0][2], line1[0][0] - line2[0][3]
        )
        dist4 = math.hypot(
            line1[0][2] - line2[0][2], line1[0][3] - line2[0][3]
        )

        if min(dist1, dist2, dist3, dist4) < 100:
            return True
        else:
            return False

    def get_line_magnitude(self, x1, y1, x2, y2):

        line_magnitude = math.sqrt(
            math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2)
        )

        return line_magnitude

    def get_distance_point_line(self, px, py, x1, y1, x2, y2):

        # http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
        LineMag = self.get_line_magnitude(x1, y1, x2, y2)

        if LineMag < 0.00000001:
            distance_point_line = 9999

            return distance_point_line

        u1 = ((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1))
        u = u1 / (LineMag * LineMag)

        if (u < 0.00001) or (u > 1):
            # // closest point does not fall within the line segment,
            # take the shorter distance
            # // to an endpoint
            ix = self.get_line_magnitude(px, py, x1, y1)
            iy = self.get_line_magnitude(px, py, x2, y2)

            if ix > iy:
                distance_point_line = iy
            else:
                distance_point_line = ix
        else:
            # Intersecting point is on the line, use the formula
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            distance_point_line = self.get_line_magnitude(px, py, ix, iy)

        return distance_point_line

    def get_distance(self, line1, line2):

        dist1 = self.get_distance_point_line(
            line1[0][0],
            line1[0][1],
            line2[0][0],
            line2[0][1],
            line2[1][0],
            line2[1][1],
        )
        dist2 = self.get_distance_point_line(
            line1[1][0],
            line1[1][1],
            line2[0][0],
            line2[0][1],
            line2[1][0],
            line2[1][1],
        )
        dist3 = self.get_distance_point_line(
            line2[0][0],
            line2[0][1],
            line1[0][0],
            line1[0][1],
            line1[1][0],
            line1[1][1],
        )
        dist4 = self.get_distance_point_line(
            line2[1][0],
            line2[1][1],
            line1[0][0],
            line1[0][1],
            line1[1][0],
            line1[1][1],
        )

        return min(dist1, dist2, dist3, dist4)

    def retrieve_polygon(
        self, the_lines: list, image: np.ndarray
    ) -> np.ndarray:
        # Since lines are usually of varying length and almost always are
        # shorter than image's height,
        # extend them first to successfully extract
        # the area confined by them
        extended_lines = list()

        # To check when just one line was detected
        # the_lines = [the_lines[1]]

        extended_lines += self.extend_lines(
            lines_to_extend=the_lines, image=image
        )

        # Once line's been extended, use them to extract the image section
        # restricted, defined by them
        support_point = extended_lines[2]
        extended_lines.append(support_point)

        points = np.array(extended_lines)

        mask = np.zeros((image.shape[0], image.shape[1]))

        # Fills  in the shape defined by the points to be white in the mask.
        # The rest is black
        cv2.fillConvexPoly(img=mask, points=points, color=1)

        # We then convert the mask into Boolean where white pixels refrecling
        # the image section we want to extract as True, the rest is False
        mask = mask.astype(np.bool)

        # Create a white empty image
        output = np.zeros_like(image)

        # Use the Boolean mask to index into the image to extract out the pixs
        # we need. All pixels that happened to be mapped as True are taken
        output[mask] = image[mask]

        output_copy = output.copy()

        # Get indices of all pixels that are black
        black_pixels_indices = np.all(output == [0, 0, 0], axis=-1)
        # Invert the matrix to get indices of not black pixels
        non_black_pixels_indices = ~black_pixels_indices

        # All black pixels become white,
        # all not black pixels get their original values
        output_copy[black_pixels_indices] = [255, 255, 255]
        output_copy[non_black_pixels_indices] = output[
            non_black_pixels_indices
        ]

        return output_copy

    def find_pole_edges(self, image):
        """Runs an image provided along the pole inclination angle
        calculating pipeline.

        :return: edges found (list of lists)
        """
        # Find all lines on the image
        raw_lines = self.generate_lines(image)

        # Rewrite lines in a proper form (x1,y1), (x2,y2) if any found.
        if raw_lines is None:
            return []

        # Process results: merge raw lines where possible to decrease the total
        # number of lines we are working with
        merged_lines = self.merge_lines(lines_to_merge=raw_lines)

        # Pick lines based on which the angle will be calculated.
        # Ideally we are looking for 2 lines which represent both pole's edges.
        # If there is 1, warn user and calculate the angle based on it.
        # Pick two opposite and parrallel lines within the merged ones.
        # We assume this is pole
        if len(merged_lines) > 1:
            the_lines = self.retrieve_pole_lines(merged_lines, image)

        elif len(merged_lines) == 1:
            print("WARNING: Only one edge detected!")
            the_lines = merged_lines

        else:
            print("WARNING: No edges detected")
            return []

        assert 1 <= len(the_lines) <= 2, "ERROR: Wrong number of lines found"

        return the_lines

    def retrieve_pole_lines(self, merged_lines, image):  # noqa
        """
        Among all lines found we need to pick only 2 - the ones
        that most likely going to pole's edges
        :param merged_lines: Lines detected (list of lists)
        :param image: image getting processed
        :return: 2 lines (list of lists)
        """
        lines_to_the_left = list()
        lines_to_the_right = list()
        left_section_and_margin = int(image.shape[1] * 0.6)
        right_section_and_margin = int(image.shape[1] * 0.4)

        if len(merged_lines) > 10:
            print(
                "WARNING: MORE THAN 10 LINES TO SORT. "
                "O(N2) WONT PROMISE YOU THAT"
            )

        while merged_lines:

            line = merged_lines.pop()
            line_angle = round(
                90
                - np.rad2deg(
                    np.arctan2(
                        abs(line[1][1] - line[0][1]),
                        abs(line[1][0] - line[0][0]),
                    )
                ),
                2,
            )
            line_lenght = math.sqrt(
                (line[1][1] - line[0][1]) ** 2 + (line[1][0] - line[0][0]) ** 2
            )

            if (
                line[0][0] <= left_section_and_margin
                and line[1][0] <= left_section_and_margin
            ):
                lines_to_the_left.append((line, line_angle, line_lenght))
                # to make sure the same line doesn't get added
                # to both subgroups if it lies in the margin
                continue

            if (
                line[0][0] >= right_section_and_margin
                and line[1][0] >= right_section_and_margin
            ):
                lines_to_the_right.append((line, line_angle, line_lenght))

        # Pick 2 best lines (2 most parallel)
        # O(n2). Slow, but we do not deal with large number of lines anyway
        optimal_lines = 180, None, None  # angle difference, line 1, line 2

        # Possible that the whole pole lies in the left part of the image
        if lines_to_the_left and not lines_to_the_right:
            # Select only among the lines to the left
            if len(lines_to_the_left) == 1:
                # Return only coordinates without angle and lenght
                return [lines_to_the_left[0][0]]

            elif len(lines_to_the_left) == 2:
                # Check if both lines to the left are relatively parallel ->
                # pole
                if abs(lines_to_the_left[0][1] - lines_to_the_left[1][1]) <= 2:
                    return [lines_to_the_left[0][0], lines_to_the_left[1][0]]
                # Else return the longest one - likely to be pole's edge
                # + some noise
                else:
                    return (
                        [lines_to_the_left[0][0]]
                        if lines_to_the_left[0][2] > lines_to_the_left[1][2]
                        else [lines_to_the_left[1][0]]
                    )

            # Have more than 2 lines to the left. Need to find the 2
            else:
                for i in range(len(lines_to_the_left) - 1):
                    for j in range(i + 1, len(lines_to_the_left)):

                        delta = abs(
                            lines_to_the_left[i][1] - lines_to_the_left[j][1]
                        )

                        if not delta < optimal_lines[0]:
                            continue
                        else:
                            optimal_lines = (
                                delta,
                                lines_to_the_left[i][0],
                                lines_to_the_left[j][0],
                            )

        # Possible that the whole pole lies in the right part of the image
        elif lines_to_the_right and not lines_to_the_left:
            # Select only among the lines to the right
            if len(lines_to_the_right) == 1:
                return [lines_to_the_right[0][0]]

            elif len(lines_to_the_right) == 2:
                # Check if both lines to the right are relatively parallel ->
                # pole
                if (
                    abs(lines_to_the_right[0][1] - lines_to_the_right[1][1])
                    <= 2
                ):
                    return [lines_to_the_right[0][0], lines_to_the_right[1][0]]
                else:
                    return (
                        [lines_to_the_right[0][0]]
                        if lines_to_the_right[0][2] > lines_to_the_right[1][2]
                        else [lines_to_the_right[1][0]]
                    )

            else:
                for i in range(len(lines_to_the_right) - 1):
                    for j in range(i + 1, len(lines_to_the_right)):

                        delta = abs(
                            lines_to_the_right[i][1] - lines_to_the_right[j][1]
                        )

                        if not delta < optimal_lines[0]:
                            continue
                        else:
                            optimal_lines = (
                                delta,
                                lines_to_the_right[i][0],
                                lines_to_the_right[j][0],
                            )

        # Ideal case - lines are to the left and to the rest.
        # Find the best 2 (most parallel ones)
        else:
            for left_line, left_angle, left_length in lines_to_the_left:
                for (
                    right_line,
                    right_angle,
                    right_length,
                ) in lines_to_the_right:

                    delta = abs(left_angle - right_angle)

                    if not delta < optimal_lines[0]:
                        continue

                    optimal_lines = delta, left_line, right_line

        return [optimal_lines[1], optimal_lines[2]]

    def generate_lines(self, image):
        """Generates lines based on which the inclination angle will be
        later calculated
        :param image: image
        :return: image with generated lines
        """
        # Resize image before
        # image = self.downsize_image(image)

        # Apply mask to remove background
        image_masked = self.apply_mask(image)

        # Generate edges
        edges = cv2.Canny(
            image_masked, threshold1=50, threshold2=200, apertureSize=3
        )
        # Based on the edges found, find lines
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=100,
            maxLineGap=100,
        )

        return lines

    def apply_mask(self, image):
        """
        Applies rectangular mask to an image in order to remove background
        and mainly focus on the pole
        :param image: original image
        :return: image with the mask applied
        """
        mask = np.zeros(image.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        # start_x, start_y, width, height
        rect = (
            int(image.shape[1] * 0.1),
            0,
            image.shape[1] - int(image.shape[1] * 0.2),
            image.shape[0],
        )

        cv2.grabCut(
            image, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT
        )

        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
        img = image * mask2[:, :, np.newaxis]

        ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)

        return thresh

    def downsize_image(self, image):
        img = imutils.resize(image, height=int(0.8 * image.shape[0]))
        return img

    def process_image(self, img: np.array) -> dict:
        """
        the main function that calls all other functions and return dict
        {'processed_img': np.array,
        'detect_angle': float}
        """
        # find edges
        result = {}
        the_edges = self.find_pole_edges(image=img)
        if the_edges:
            # Calculate angle
            the_angle = calculate_angle(the_lines=the_edges)
            processed_image = draw_lines_write_text(
                lines=the_edges, image=img, angle=the_angle
            )
            result["processed_img"] = processed_image
            result["angle"] = the_angle
        return result
