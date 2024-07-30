import time
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


def plot_polygon_from_coords(coords):
    coords = np.array(coords)
    plt.plot(coords[:, 0], coords[:, 1], 'r-')
    plt.plot(coords[0, 0], coords[0, 1], 'ro')
    plt.gca().set_aspect('equal')
    plt.show()



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


def decompose_rectilinear_polygon(polygon):
    exterior_coords = np.array(polygon.exterior.coords)
    interior_coords = [np.array(interior.coords) for interior in polygon.interiors]

    all_coords = np.vstack([exterior_coords] + interior_coords)
    x_coords = sorted(set(all_coords[:, 0]))
    y_coords = sorted(set(all_coords[:, 1]))

    rectangles = []

    for i in range(len(x_coords) - 1):
        for j in range(len(y_coords) - 1):
            rect = Polygon([
                (x_coords[i], y_coords[j]),
                (x_coords[i + 1], y_coords[j]),
                (x_coords[i + 1], y_coords[j + 1]),
                (x_coords[i], y_coords[j + 1])
            ])
            if polygon.contains(rect):
                rectangles.append(rect)
            elif polygon.intersects(rect):
                intersection = polygon.intersection(rect)
                if intersection.area > 0:
                    if intersection.geom_type == 'Polygon':
                        rectangles.append(intersection)
                    elif intersection.geom_type == 'MultiPolygon':
                        rectangles.extend(list(intersection.geoms))

    return rectangles

def resolve_overlaps(rectangles):
    boxes = [box(rect[0][0], rect[1][0], rect[0][1], rect[1][1]) for rect in rectangles]

    union = unary_union(boxes)
    result_boxes = []
    if isinstance(union, MultiPolygon):
        for part in union.geoms:
            if is_polygon_box(part):
                result_boxes.append(part)
            else:
                result_boxes.extend(decompose_rectilinear_polygon(part))
    else:
        if is_polygon_box(union):
            result_boxes.append(union)
        else:
            result_boxes.extend(decompose_rectilinear_polygon(union))

    return [[[box.bounds[0], box.bounds[2]], [box.bounds[1], box.bounds[3]]] for box in result_boxes]


def add_rectangle_to_decomposed(existing_rectangles, new_rectangle):
    new_box = box(new_rectangle[0][0], new_rectangle[1][0], new_rectangle[0][1], new_rectangle[1][1])
    intersecting_rectangles = []
    non_intersecting_rectangles = []

    for rect in existing_rectangles:
        existing_box = box(rect[0][0], rect[1][0], rect[0][1], rect[1][1])
        if new_box.intersects(existing_box):
            intersecting_rectangles.append(existing_box)
        else:
            non_intersecting_rectangles.append(rect)

    if intersecting_rectangles:
        # Only decompose the intersecting area
        union = unary_union([new_box] + intersecting_rectangles)
        decomposed = decompose_rectilinear_polygon(union)
        result_boxes = [shapely_box_to_rect(box) for box in decomposed]
        return non_intersecting_rectangles + result_boxes
    else:
        # If no intersections, simply add the new rectangle
        return existing_rectangles + [new_rectangle]



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
        "Hole": [
            [[1, 5], [1, 2]],
            [[1, 5], [3, 4]],
            [[1, 2], [1, 4]],
            [[3, 4], [1, 4]],
            [[2.5, 3.5], [2.5, 3.5]]
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

    # TODO: optimizations:
        # 1. adding one polygon to existing list
