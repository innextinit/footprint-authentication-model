import os
import random
import requests

# Function to authenticate an image using the /access endpoint
def authenticate_image(file_path, endpoint_url):
    # Open the image file
    with open(file_path, 'rb') as file:
        files = {'file': file}
        # Send a POST request to the /access endpoint
        response = requests.post(endpoint_url, files=files)
        # Console the result output
        print(response.text)

def authenticate_image(file_path, endpoint_url):
    # Open the image file
    with open(file_path, 'rb') as file:
        files = {'file': file}
        # Send a POST request to the /access endpoint
        response = requests.post(endpoint_url, files=files)
        # Prepare the match result string
        match_result = f"filename: {os.path.basename(file_path)}, match_response: {response.text}\n"
        # Append the match result to the file
        with open("match_result.txt", "a") as match_file:
            match_file.write(match_result)

# Function to authenticate 100 random images from the dataset folder
def authenticate_images(endpoint_url):
    # Directory containing dataset images
    current_directory = os.path.dirname(os.path.realpath(__file__))
    dataset_folder = os.path.join(current_directory, 'dataset/footprints')
    
    # Get a list of all image files in the dataset folder
    image_files = [os.path.join(dataset_folder, file_name) for file_name in os.listdir(dataset_folder)
                   if os.path.isfile(os.path.join(dataset_folder, file_name)) and
                   file_name.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Randomly select 100 images from the dataset
    random_images = random.sample(image_files, 100)
    
    # Iterate over each randomly selected image
    for file_path in random_images:
        # Authenticate the image using the authenticate_image function
        authenticate_image(file_path, endpoint_url)

# Main function
def main():
    # URL of the /access endpoint
    access_endpoint_url = 'http://localhost:5000/access'  # Update the URL with your actual endpoint URL
    
    # Authenticate 100 random images using the authenticate_images function
    authenticate_images(access_endpoint_url)

if __name__ == '__main__':
    main()
