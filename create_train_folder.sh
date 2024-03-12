#!/bin/bash

# Create directories and copy images for each user
for ((i=14; i<=95; i++)); do
    # Create the user directory
    mkdir -p "train/user$i" "test/user$i"
    
    # Copy images into the user directory
    cp "dataset/footprints/0${i}_1.jpg" "train/user$i/"
    cp "dataset/footprints/0${i}_2.jpg" "train/user$i/"
    cp "dataset/footprints/0${i}_3.jpg" "train/user$i/"
    cp "dataset/footprints/0${i}_4.jpg" "train/user$i/"
    cp "dataset/footprints/0${i}_5.jpg" "test/user$i/"
    cp "dataset/footprints/0${i}_6.jpg" "validation/"
done

