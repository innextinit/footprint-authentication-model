from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker 

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

    def __repr__(self):
        return f"<Footprint(id={self.id}, image_name='{self.image_name}', weight='{self.weight}')>"

# Create tables in the database if they don't exist
Base.metadata.create_all(engine)

def save_features(image_name, prediction_weight):
    new_footprint = Footprint(image_name=image_name, weight=prediction_weight)

    session.add(new_footprint)
    session.commit()

    return

def find_matching_features(image_name, prediction_weight):
    # Query all footprints from the database
    all_footprints = session.query(Footprint).all()

    matching_indices = []
    min_diff = float('inf')  # Initialize minimum difference to infinity
    best_match = None  # Initialize best match to None
    accuracy = 0 # Initialize to 0%

    for stored_footprint in all_footprints:
        stored_weight = float(stored_footprint.weight)

        diff = abs(prediction_weight - stored_weight)

        if diff < min_diff:
            min_diff = diff
            accuracy = 100 - (diff / stored_weight) * 100
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
