from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker 
from scipy.spatial.distance import euclidean
import numpy as np

# Create the SQLAlchemy database engine
engine = create_engine('sqlite:///footprint_database.db', echo=True)  # Change database URL as needed

# Create a session to interact with the database
Session = sessionmaker(bind=engine)
session = Session()

Base = declarative_base()

# Define a Footprint model to store image names and features in the database
class Footprint(Base):
    __tablename__ = 'footprints'

    id = Column(Integer, primary_key=True)
    image_name = Column(String)
    features = Column(String)

    def __repr__(self):
        return f"<Footprint(id={self.id}, image_name='{self.image_name}', features='{self.features}')>"

# Create tables in the database if they don't exist
Base.metadata.create_all(engine)

# Function to save extracted features to the database
def save_features(image_name, features):
    features_array = np.array(features).flatten()
    features_bytes = features_array.tobytes()

    new_footprint = Footprint(image_name=image_name, features=features_bytes)

    # Add the new_footprint to the session and commit it to the database
    session.add(new_footprint)
    session.commit()

    print(f"Features extracted from '{image_name}' saved to the database.")

# Function to find matching features in the database

def find_matching_features(features):
    # Convert the extracted features to a NumPy array for comparison
    features_array = np.array(features).flatten()
    print("features_array", features_array)

    # Query all footprints from the database
    all_footprints = session.query(Footprint).all()

    matching_indices = []
    for index, footprint in enumerate(all_footprints):
        # Convert stored features from string to NumPy array for comparison
        stored_features_bytes = footprint.features
        stored_features_array = np.frombuffer(stored_features_bytes, dtype=np.float32)

        # Calculate similarity using Euclidean distance
        distance = euclidean(features_array, stored_features_array)
        print("distance", distance)
        
        # Define a threshold for similarity comparison
        threshold = 0.1  # 90% similarity

        if distance < threshold:
            matching_indices.append((footprint.id, footprint.image_name, index))

    return matching_indices


