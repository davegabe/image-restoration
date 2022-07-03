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


def draw(img: cv.Mat, DEBUG=False):
    """
        Draws a random line on the image.
        Args:
            img: The image to draw the line on.
            DEBUG: Whether to open the image in a window.
    """
    x = img.shape[0]
    y = img.shape[1]
    minPoints = 3
    maxPoints = 10

    # list of random points to draw
    points = [(random.randint(0, x), random.randint(0, y))
              for i in range(minPoints, maxPoints)]

    # draw line passing through points
    for i in range(1, len(points)):
        cv.line(img, points[i-1], points[i], 0, int(0.035 * min(x, y))) # thickness of line based on image size

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
