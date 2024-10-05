from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import requests
import time
import os
import base64
from PIL import Image

from selenium import webdriver
from selenium.webdriver.chrome.service import Service

# Start the Chrome browser
driver = webdriver.Chrome()

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

def get_image_urls(query, number_of_links, sleep_time=2):

    search_url = f"https://www.google.com/search?q={query}&tbm=isch"
    
    # Open the Google Images page for the search query
    driver.get(search_url)
    image_urls = set()

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

def download_images(image_urls, folder_path, min_height, min_width, time_sleep = 2):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for i, img_url in enumerate(image_urls):
        try:
            time.sleep(time_sleep) 
            if "http" in img_url:
                download_image(img_url, os.path.join(folder_path, f"{i+1}.jpg"))
            elif "base64" in img_url:
                base64_to_image(img_url, os.path.join(folder_path, f"{i+1}.jpg"))
            check_image_quality(os.path.join(folder_path, f"{i+1}.jpg"), min_width, min_height)

        except Exception as e:
            print(f"Failed to download image {i+1} due to {e}")

labels = ["Stop Sign", "Yield Sign", "Speed Limit Sign", "Pedestrian Crossing Sign", "No Entry Sign"]
max_images = 2000
min_height = 100
min_width = 100

for label in labels:
    image_urls = get_image_urls(label, max_images)
    download_images(image_urls, f"./images/{label}", min_height=min_height, min_width=min_width)

# Close the browser session
driver.quit()