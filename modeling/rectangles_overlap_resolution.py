import time
from copy import deepcopy

import numpy as np
from shapely import MultiPolygon
from shapely.geometry import box, Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt


def rect_to_shaply_box(rect):
    minx, maxx = rect[0]
    miny, maxy = rect[1]
    return box(minx, miny, maxx, maxy)


def shapely_box_to_rect(box):
    return [[box.bounds[0], box.bounds[2]], [box.bounds[1], box.bounds[3]]]


def is_polygon_box(polygon):
    if not isinstance(polygon, Polygon) or len(polygon.exterior.coords) != 5:
        return False  # Ensures exactly four vertices plus a closing point

    coords = np.array(polygon.exterior.coords)

    # Calculating distances for opposite sides using NumPy's norm function
    side1 = np.linalg.norm(coords[0] - coords[1])
    side2 = np.linalg.norm(coords[1] - coords[2])
    side3 = np.linalg.norm(coords[2] - coords[3])
    side4 = np.linalg.norm(coords[3] - coords[0])

    # Check for equal length of opposite sides
    return np.isclose(side1, side3) and np.isclose(side2, side4)


def decompose_to_rectangles(polygon):
    curr_polygon = deepcopy(polygon)
    rectangles = []

    while curr_polygon.area > 0 and not is_polygon_box(curr_polygon):
        # Decompose the polygon to rectangles
        curr_polygon_vertices = list(curr_polygon.exterior.coords)
        curr_polygon_vertices = curr_polygon_vertices[:-1]  # remove the last point which is the same as the first one

        # sort the vertices by x, then by y:
        curr_polygon_vertices.sort(key=lambda x: (x[0], x[1]), reverse=False)

        # take the leftmost point
        left_down = curr_polygon_vertices.pop(0)
        # there should be another point with the same x coordinate
        left_up = curr_polygon_vertices.pop(0)

        # find second leftmost point, has to be different than left_up x coordinate
        for i, vertex in enumerate(curr_polygon_vertices):
            if vertex[0] != left_up[0]:
                right_down = curr_polygon_vertices.pop(i)
                break

        # those three points define a rectangle
        right_up = [right_down[0], left_up[1]]
        rectangles.append(box(left_down[0], left_down[1], right_up[0], right_up[1]))

        curr_polygon = curr_polygon.difference(rectangles[-1])

    if curr_polygon.area > 0:
        rectangles.append(curr_polygon)

    return rectangles


def plot_polygon_from_coords(coords):
    coords = np.array(coords)
    plt.plot(coords[:, 0], coords[:, 1], 'r-')
    plt.plot(coords[0, 0], coords[0, 1], 'ro')
    plt.gca().set_aspect('equal')
    plt.show()


def add_box_to_merged_list(merged_boxes, box_to_add):
    new_boxes = [box_to_add]
    final_boxes_to_add = []

    while new_boxes:
        current_box = new_boxes.pop(0)
        any_intersection = False

        for merged_box in merged_boxes:
            if merged_box.intersects(current_box):
                intersection = merged_box.intersection(current_box)
                if not isinstance(intersection, Polygon):
                    # that's point or line, which is not really an intersection
                    continue

                any_intersection = True
                difference = current_box.difference(intersection)

                if isinstance(difference, Polygon):
                    if is_polygon_box(difference):
                        new_boxes.append(difference)
                    else:
                        new_boxes.extend(decompose_to_rectangles(difference))

                if isinstance(difference, MultiPolygon):
                    for part in difference.geoms:
                        if is_polygon_box(part):
                            new_boxes.append(part)
                        else:
                            new_boxes.extend(decompose_to_rectangles(part))

                break  # we already processed this one and it got decomposed

        if not any_intersection:
            final_boxes_to_add.append(current_box)

    merged_boxes.extend(final_boxes_to_add)


def resolve_overlaps(rectangles):
    boxes = [rect_to_shaply_box(rect) for rect in rectangles]
    merged_boxes = [boxes[0]]

    for box in boxes[1:]:
        add_box_to_merged_list(merged_boxes, box)

    return [shapely_box_to_rect(box) for box in merged_boxes]


def plot_orig_and_resolved(orig, resolved, xlim, ylim):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    for rect in orig:
        axs[0].add_patch(plt.Rectangle((rect[0][0], rect[1][0]), rect[0][1] - rect[0][0], rect[1][1] - rect[1][0],
                                       fill=True, color='blue', alpha=0.5))

    for rect in resolved:
        axs[1].add_patch(plt.Rectangle((rect[0][0], rect[1][0]), rect[0][1] - rect[0][0], rect[1][1] - rect[1][0],
                                       fill=True, color='red', alpha=0.5))

    for ax in axs:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')

    plt.show()


if __name__ == '__main__':
    test_cases = {
        "Single Overlap": [
            [[1, 4], [1, 4]],
            [[3, 6], [3, 6]]
        ],
        "Multiple Overlaps": [
            [[1, 5], [1, 3]],
            [[2, 6], [2, 5]],
            [[3, 7], [1, 4]]
        ],
        "L-Shape Overlap": [
            [[1, 3], [1, 5]],
            [[2, 5], [3, 4]]
        ],
        "Complete Overlap": [
            [[2, 5], [2, 5]],
            [[3, 4], [3, 4]]  # Completely inside the first rectangle
        ],
        "Non-overlapping": [
            [[1, 2], [1, 2]],
            [[3, 4], [3, 4]],
            [[5, 6], [5, 6]]
        ],
        "T-Shaped Overlap": [
            [[2, 6], [3, 4]],
            [[4, 5], [1, 5]]
        ],
        "Five Rectangle Overlap": [
            [[1, 3], [1, 4]],
            [[2, 4], [3, 5]],
            [[3, 5], [1, 2]],
            [[1, 2], [2, 3]],
            [[2, 3], [4, 6]]
        ],
        "Ten Rectangle Overlap": [
            [[1, 2], [1, 3]],
            [[1, 3], [2, 4]],
            [[2, 4], [1, 3]],
            [[3, 5], [2, 4]],
            [[3, 4], [3, 5]],
            [[4, 6], [1, 3]],
            [[5, 7], [2, 4]],
            [[5, 7], [1, 2]],
            [[6, 8], [2, 5]],
            [[7, 9], [3, 5]]
        ],
        "custom test case": [
            [[-0.85, -0.77], [-0.73, -0.65]],
            [[-0.89, -0.81], [-0.75, -0.67]]
        ]
    }

    # Process each test case
    for name, rects in test_cases.items():
        start_time = time.time()
        resolved_rectangles = resolve_overlaps(rects)
        # resolved_rectangles = rects
        end_time = time.time()

        print(f"Test: {name}")
        print(f"Runtime: {end_time - start_time:.6f} seconds")
        x_lim = [min(rect[0][0] for rect in rects) - 0.5, max(rect[0][1] for rect in rects) + 0.5]
        y_lim = [min(rect[1][0] for rect in rects) - 0.5, max(rect[1][1] for rect in rects) + 0.5]
        plot_orig_and_resolved(rects, resolved_rectangles, x_lim, y_lim)
