from icrawler.builtin import GoogleImageCrawler
import os
import cv2
import numpy as np

def download(training_path, evaluation_path, width, height, keyword, number):
    """
    Downloads training data from the internet and saves it to the specified destination, separated into training and evaluation data.
    
    Args:
        training_path: The path to save the training data.
        evaluation_path: The path to save the evaluation data.
    """
    
    keyword = keyword+" imagesize:" + str(width) + "x" + str(height)

    google_crawler = GoogleImageCrawler(
        feeder_threads=1,
        parser_threads=2,
        downloader_threads=4,
        storage={'root_dir': evaluation_path})
       
        # ma visto che tanto, se io muovo le immagini dalla cartella images a training, il 20% rimane in images, a sto punto metto tutto dentro eval e poi mi prendo l'80 e via

    google_crawler.crawl(keyword=keyword, max_num=number, file_idx_offset=0)

    tmp_img = np.zeros((1, 1, 3), dtype="uint8")
    files = os.listdir(evaluation_path)

    for i in range(int(number*0.8)):    #prendo l'80% delle immagini
        old_file_path = os.path.join(evaluation_path, files[i])
        new_file_path = os.path.join(training_path, files[i])
        if not (files[i].startswith('.')):  # visto che tra i file conta pure .DS_Store, il che mi fa sfarfallare tutto, controllo che non inizi in maniera strana
        # per usare rename devo prima creare un file con lo stesso nome nella destinazione
            cv2.imwrite(new_file_path, tmp_img)
            os.rename(old_file_path, new_file_path)
    
    print(training_path, evaluation_path)
    pass
