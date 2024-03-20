#Load libraries
import os
import torch
import glob
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision.transforms import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

current_directory = os.path.dirname(os.path.realpath(__file__))
train_path = os.path.join(current_directory, '..', 'train')
test_path = os.path.join(current_directory, '..', 'test')

predictions = {}

# Get the directories inside the train_path
directories = os.listdir(train_path)
# Filter out the .DS_Store file if present
directories = [d for d in directories if os.path.isdir(os.path.join(train_path, d)) and d != '.DS_Store']
classes = sorted(directories)

#CNN Network
class ConvNet(nn.Module):
    def __init__(self,num_classes=95): # always update num_classes to the number of training set
        super(ConvNet,self).__init__()
        
        #Output size after convolution filter
        #((w-f+2P)/s) +1
        
        #Input shape= (256,3,150,150)
        
        self.conv1=nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,stride=1,padding=1)
        #Shape= (256,12,150,150)
        self.bn1=nn.BatchNorm2d(num_features=12)
        #Shape= (256,12,150,150)
        self.relu1=nn.ReLU()
        #Shape= (256,12,150,150)
        
        self.pool=nn.MaxPool2d(kernel_size=2)
        #Reduce the image size be factor 2
        #Shape= (256,12,75,75)
        
        
        self.conv2=nn.Conv2d(in_channels=12,out_channels=20,kernel_size=3,stride=1,padding=1)
        #Shape= (256,20,75,75)
        self.relu2=nn.ReLU()
        #Shape= (256,20,75,75)
        
        self.conv3=nn.Conv2d(in_channels=20,out_channels=32,kernel_size=3,stride=1,padding=1)
        #Shape= (256,32,75,75)
        self.bn3=nn.BatchNorm2d(num_features=32)
        #Shape= (256,32,75,75)
        self.relu3=nn.ReLU()
        #Shape= (256,32,75,75)
        
        self.fc=nn.Linear(in_features=75 * 75 * 32,out_features=num_classes)
        
        #Feed forward function
    def forward(self,input):
        output=self.conv1(input)
        output=self.bn1(output)
        output=self.relu1(output)
        output=self.pool(output)
        
        output=self.conv2(output)
        output=self.relu2(output)
        
        output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3(output)
        
        output=output.view(-1,32*75*75)
        output=self.fc(output)
        return output

# Training Function
def train_model(train_loader, num_epochs=20, learning_rate=0.001):
    model = ConvNet(num_classes=95)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for i, (images, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute statistics
            running_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
        
        # Compute epoch accuracy
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_predictions

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

    return model

# Data Loader
def get_data_loader(train_path, batch_size=256):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((150,150)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = torchvision.datasets.ImageFolder(train_path, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return train_loader

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)

def load_model(filepath):
    model = ConvNet(num_classes=95)
    model.load_state_dict(torch.load(filepath))
    model.eval()
    return model

#Transforms
transformer = transforms.Compose([
    transforms.Resize((150,150)),
    transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
    transforms.Normalize(mean=[0.5,0.5,0.5], # 0-1 to [-1,1] , formula (x-mean)/std
                        std=[0.5,0.5,0.5])
])

def prediction(loaded_model, img_path, true_labels, predicted_labels):
    # Open the image file
    image = Image.open(img_path)
    image = image.convert("RGB")

    transformed_image = transformer(image)
    # Convert the image to a tensor and add a batch dimension
    image_tensor = torch.unsqueeze(transformed_image, 0)

    # Make prediction
    loaded_model.eval()
    with torch.no_grad():
        outputs = loaded_model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    
    # Convert prediction index to class label
    pred_class = classes[predicted.item()]

     # Append true label and prediction for evaluation
    true_label = img_path.split(os.path.sep)[-2]  # Assuming the label is the folder name
    true_labels.append(true_label)
    predicted_labels.append(pred_class)
    
    return pred_class

def load_prediction(image_files, loaded_model, true_labels, predicted_labels):
    for image_file in image_files:
        # Get the filename
        filename = os.path.basename(image_file)
        # Make prediction
        pred = prediction(loaded_model, image_file, true_labels, predicted_labels)
        # Store prediction in the dictionary
        predictions[filename] = pred
    return predictions

def load_images():
    image_files = []
    # Iterate over each subfolder in the test folder
    for folder_name in os.listdir(test_path):
        folder_path = os.path.join(test_path, folder_name)
        if os.path.isdir(folder_path):
            # Search for image files within the subfolder and append their paths to the list
            image_files.extend(glob.glob(os.path.join(folder_path, '*.jpg')))
    return image_files

if __name__ == "__main__":
    true_labels = []
    predicted_labels = []
    train_loader = get_data_loader(train_path)
    trained_model = train_model(train_loader)
    save_model(trained_model, "tt_trained_model.pth")
    loaded_model = load_model("tt_trained_model.pth")
    image_files = load_images()
    result = load_prediction(image_files, loaded_model, true_labels, predicted_labels)
    print("result",result)
    # Compute evaluation metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='micro')
    recall = recall_score(true_labels, predicted_labels, average='micro')
    f1 = f1_score(true_labels, predicted_labels, average='micro')
    confusion_mat = confusion_matrix(true_labels, predicted_labels)
    confusion_mat_str = "\n".join([str(row) for row in confusion_mat])

    # Write evaluation metrics to file
    with open("evaluation_metrics.txt", "a") as metrics_file:
        metrics_file.write(f"Accuracy: {accuracy}\n")
        metrics_file.write(f"Precision: {precision}\n")
        metrics_file.write(f"Recall: {recall}\n")
        metrics_file.write(f"F1 Score: {f1}\n")
        metrics_file.write(f"True labels: {true_labels}\n")
        metrics_file.write(f"Predicted labels: {predicted_labels}\n")
        metrics_file.write(f"Confusion Matrix:\n{confusion_mat_str}\n\n")
    
    