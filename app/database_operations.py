from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import numpy as np
from scipy.spatial.distance import cosine
import pickle

def compare_images(image1_data, image2_data):
    """
    Compare two images based on their numpy array representations.
    """
    # Calculate the cosine similarity between the two images
    similarity = 1 - cosine(image1_data.flatten(), image2_data.flatten())

    return similarity



# Create the SQLAlchemy database engine
engine = create_engine('sqlite:///footprint_database.db', echo=True)

# Create a session to interact with the database
Session = sessionmaker(bind=engine)
session = Session()

Base = declarative_base()

# Define a Footprint model
class Footprint(Base):
    __tablename__ = 'footprints'

    id = Column(Integer, primary_key=True)
    image_name = Column(String)
    weight = Column(String)
    image_data = Column(String)

    def __repr__(self):
        return f"<Footprint(id={self.id}, image_name='{self.image_name}', weight='{self.weight}')>"

# Create tables in the database if they don't exist
Base.metadata.create_all(engine)

def save_features(image_name, prediction_weight, processed_image):
    # Serialize the numpy array
    processed_image_bytes = pickle.dumps(processed_image)

    new_footprint = Footprint(image_name=image_name, weight=prediction_weight, image_data=processed_image_bytes)

    session.add(new_footprint)
    session.commit()

    return

def find_matching_features(image_name, prediction_weight, processed_image):
    # Query all footprints from the database
    all_footprints = session.query(Footprint).all()

    matching_indices = []
    pre_diff = 0
    best_match = None
    accuracy = 0

    for stored_footprint in all_footprints:
        # Deserialize the stored numpy array
        stored_image_data = pickle.loads(stored_footprint.image_data)

        diff = compare_images(processed_image, stored_image_data)

        if diff > pre_diff:
            pre_diff = diff
            accuracy = diff * 100
            best_match = stored_footprint
    
    if accuracy < 90:
        return None

    if best_match is not None:
        matching_indices.append({
            'id': best_match.id,
            'image_name': best_match.image_name,
            'accuracy': accuracy,
            'matching_image_name': image_name
        })

    return matching_indices
