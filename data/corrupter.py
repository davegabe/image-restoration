import random
import cv2 as cv
import os


def corrupt(training_path: str, evaluation_path: str, training_corrupted_path: str, evaluation_corrupted_path: str, augmentation: int = 0):
    """
    Corrupt training and evaluation data and save it to the specified destinations.

    Args:
        training_path: The path to the training data.
        evaluation_path: The path to the evaluation data.
        training_corrupted_path: The path to save the corrupted training data.
        evaluation_corrupted_path: The path to save the corrupted evaluation data.
    """
    # for each file in the training data, corrupt it and save it to the training corrupted path
    for file_name in os.listdir(training_path):
        for i in range(augmentation):
            img = cv.imread(training_path + file_name)
            draw(img)

            new_name = f"{os.path.splitext(file_name)[0]}_corrupted_{str(i)}{os.path.splitext(file_name)[1]}"

            final_path = os.path.join(training_corrupted_path, new_name)

            cv.imwrite(final_path, img)

    # for each file in the evaluation data, corrupt it and save it to the evaluation corrupted_path
    for file_name in os.listdir(evaluation_path):
        for i in range(augmentation):  # repeat the process for the specified number of times
            img = cv.imread(evaluation_path + file_name)
            draw(img)

            new_name = f"{os.path.splitext(file_name)[0]}_corrupted_{str(i)}{os.path.splitext(file_name)[1]}"

            final_path = os.path.join(evaluation_corrupted_path, new_name)

            cv.imwrite(final_path, img)


def drawCurve(img: cv.Mat, start_point: tuple):
    """
        Draws a curve on the image starting from one point and moving to near points on the image.

        Args:
            img: The image to draw the curve on.
            DEBUG: Whether or not to print debug messages.
    """
    x = img.shape[0]
    y = img.shape[1]
    thickness = int(0.035 * min(x, y))
    points = [start_point]
    minPoints = 20
    maxPoints = 50

    # random direction tuple in [-1, 1]
    direction = ((random.random()-0.5)*2, (random.random()-0.5)*2)
    # list of random points to draw
    for i in range(minPoints, maxPoints):
        # get a random point near the last point in the list
        x1 = int(points[-1][0] + direction[0] * thickness)
        y1 = int(points[-1][1] + direction[1] * thickness)

        # clamp the point to the image
        x1 = max(0, min(x1, x-1))
        y1 = max(0, min(y1, y-1))

        # get next direction "human-style" (trying to not completely inverse it)
        while True:
            new_direction = ((random.random()-0.5)*2, (random.random()-0.5)*2) # get a new random direction in [-1, 1]
            if abs(new_direction[0]-direction[0]) < 0.7 and abs(new_direction[1]-direction[1]) < 0.7: 
                direction = new_direction
                break
        
        # add the point to the list
        points.append((x1, y1))

    # draw line passing through points
    for i in range(1, len(points)):
        cv.line(img, points[i-1], points[i], 0, thickness)


def draw(img: cv.Mat, DEBUG=False):
    """
        Draws a random line on the image.
        Args:
            img: The image to draw the line on.
            DEBUG: Whether to open the image in a window.
    """
    x = img.shape[0]
    y = img.shape[1]
    minCurves = 1
    maxCurves = 4

    # draw random number of curves
    for i in range(random.randint(minCurves, maxCurves)):
        drawCurve(img, (random.randint(0, x), random.randint(0, y)))

    if (DEBUG == True):
        cv.imshow("img", img)
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == "__main__":
    while True:
        training_path = "./Dataset/Training/Original/"
        # list files inside training_path
        training_files = os.listdir(training_path)
        # imread first file in training_files
        img = cv.imread(
            training_path + training_files[random.randint(0, len(training_files)-1)])
        # draw line passing through points
        draw(img, DEBUG=True)
