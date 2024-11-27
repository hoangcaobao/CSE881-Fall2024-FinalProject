from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import requests
import time
import os
import base64
from PIL import Image
import pickle
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from multiprocessing import Pool
import multiprocessing

def base64_to_image(url, file_path):
    prefix = "base64,"
    base64_img = url[url.find(prefix) + len(prefix):]

    image_data = base64.b64decode(base64_img)
    with open(file_path, "wb") as file:
        file.write(image_data)

def download_image(url, file_path):
    response = requests.get(url)
    with open(file_path, "wb") as f:
        f.write(response.content)

def check_image_quality(file_path, min_width, min_height):
    try:
        with Image.open(file_path) as img:
            width, height = img.size
            if width < min_width or height < min_height:
                os.remove(file_path)  # Remove if below threshold
                print(f"Removed {file_path} due to insufficient quality.")
    except Exception as e:
        print(f"Failed to check quality of {file_path} due to {e}")
        if os.path.exists(file_path):
            os.remove(file_path)

def process_image_download(img_url, folder_path, i, min_width, min_height, time_sleep):
    time.sleep(time_sleep)
    file_path = os.path.join(folder_path, f"{i+1}.jpg")
    try:
        if "http" in img_url:
            download_image(img_url, file_path)
        elif "base64" in img_url:
            base64_to_image(img_url, file_path)
        check_image_quality(file_path, min_width, min_height)
    except Exception as e:
        print(f"Failed to download image {i+1} due to {e}")

def download_images_multiprocess(image_urls, folder_path, min_height, min_width, time_sleep=2, num_workers=5):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Prepare arguments for each task
    args = [
        (img_url, folder_path, i, min_width, min_height, time_sleep)
        for i, img_url in enumerate(image_urls)
    ]
    
    # Use Pool for multiprocessing
    with Pool(processes=num_workers) as pool:
        pool.starmap(process_image_download, args)
        
def get_image_urls(query, number_of_links, sleep_time=1):
    on_road = query + " On Road"
    in_city = query + " In City"
    search_urls = [f"https://www.google.com/search?q={query}&tbm=isch", f"https://www.google.com/search?q={on_road}&tbm=isch", f"https://www.google.com/search?q={in_city}&tbm=isch", f"https://www.google.com/search?q={query}&udm=28"]
    
    image_urls = set()
    for search_url in search_urls:
        # Open the Google Images page for the search query
        driver.get(search_url)
        
        iteration = 0
        
        while len(image_urls) < number_of_links and iteration < 20:
            iteration += 1
            time.sleep(sleep_time)
            soup = BeautifulSoup(driver.page_source, "html.parser")

            # Find all the image elements on the page
            image_elements = soup.find_all("img")

            # Extract image URLs from the image elements
            for img in image_elements:
                if(len(image_urls) > number_of_links):
                    break
                img_url = img.get("src")
                img_height = img.get("height")
                img_width = img.get("width")

                # Avoid too small image
                if (img_height and int(img_height) <= 20) or (img_width and int(img_width) <= 20):
                    continue
                
                if img_url and img_url not in image_urls:
                    image_urls.add(img_url)

            # Scroll the page
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    return image_urls
if __name__ == "__main__":
    labels = ["Stop Sign", "Speed Limit Sign", "Crosswalk Sign", "Traffic Light"]

    max_images = 5000
    min_height = 100
    min_width = 100

    count_worker = multiprocessing.cpu_count()

    if not os.path.exists("image_urls_all.pkl"):
        # Start the Chrome browser
        driver = webdriver.Chrome()

        image_urls_all = []

        for label in labels:
            image_urls = get_image_urls(label, max_images)
            image_urls_all.append(image_urls)
            print(len(image_urls))

        with open("image_urls_all.pkl", "wb") as f:
            pickle.dump(image_urls_all, f)

        # Close the browser session
        driver.quit()

    with open("image_urls_all.pkl", "rb") as f:
        image_urls_all = pickle.load(f)

    for i in range(len(labels)):
        download_images_multiprocess(image_urls_all[i], f"./images/{labels[i]}", min_height=min_height, min_width=min_width, num_workers=count_worker)
