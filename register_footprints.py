import os
import requests

# Function to register an image using the /reg endpoint
def register_image(file_path, endpoint_url):
    # Open the image file
    with open(file_path, 'rb') as file:
        files = {'file': file}
        # Send a POST request to the /reg endpoint
        response = requests.post(endpoint_url, files=files)
        if response.status_code == 200:
            print(f"Image {file_path} registered successfully")
        else:
            print(f"Failed to register image {file_path}. Error: {response.text}")

# Main function
def main():
    # Directory containing validation images
    current_directory = os.path.dirname(os.path.realpath(__file__))
    validation_folder = os.path.join(current_directory, 'validation')
    # URL of the /reg endpoint
    reg_endpoint_url = 'http://localhost:5000/reg'  # Update the URL with your actual endpoint URL
    
    # Iterate over each file in the validation folder
    for file_name in os.listdir(validation_folder):
        file_path = os.path.join(validation_folder, file_name)
        # Check if the file is a valid image file
        if os.path.isfile(file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Register the image using the register_image function
            register_image(file_path, reg_endpoint_url)

if __name__ == '__main__':
    main()
