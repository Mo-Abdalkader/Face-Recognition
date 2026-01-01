"""
Example: Add Person to Database
Shows how to register a new person with multiple face images
"""

from config import Config as CFG

# Example function to add a person with multiple images
def add_person_to_database(
        face_id: str,
        name: str,
        image_paths: list,
        department: str = None,
        role: str = None,
        email: str = None,
        phone: str = None,
        notes: str = None
):
    """
    Add a person to the database with multiple face images

    Args:
        face_id: Unique identifier (e.g., 'EMP001')
        name: Person's name
        image_paths: List of paths to face images
        department: Department name
        role: Job title
        email: Email address
        phone: Phone number
        notes: Additional notes
    """
    from database.face_database import FaceDatabase  # commented for Kaggle

    # Initialize database
    db = FaceDatabase(CFG.DATABASE_PATH)

    # Add person metadata
    success = db.add_person(
        face_id=face_id,
        name=name,
        department=department,
        role=role,
        email=email,
        phone=phone,
        notes=notes
    )

    if not success:
        print(f"Failed to add person {name}")
        db.close()
        return False

    # Generate and add embeddings for all images
    print(f"\nGenerating embeddings for {len(image_paths)} images...")

    for img_path in image_paths:
        print(f"  Processing: {img_path}")

        # Generate embedding
        embedding = embedding_generator.generate_embedding(img_path)

        if embedding is not None:
            # Add to database
            db.add_embedding(face_id, embedding, img_path)
        else:
            print(f"  Warning: Failed to generate embedding for {img_path}")

    db.close()

    print(f"\n✓ Successfully added {name} (ID: {face_id})")
    print("Remember to reload the recognizer database:")
    print("  recognizer.reload_database()")

    return True

# Example usage:
# add_person_to_database(
#     face_id='EMP001',
#     name='Ahmed Hassan',
#     image_paths=[
#         '/path/to/ahmed_front.jpg',
#         '/path/to/ahmed_side.jpg',
#         '/path/to/ahmed_smile.jpg'
#     ],
#     department='Engineering',
#     role='Software Engineer',
#     email='ahmed@company.com',
#     phone='+20-123-456-7890'
# )