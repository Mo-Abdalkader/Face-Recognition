"""
Batch Face Recognition Utility
Process multiple images and generate recognition report
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm


def batch_recognize_faces(
        image_dir: str,
        recognizer: object,
        output_csv: str = 'recognition_results.csv'
):
    """
    Recognize faces in all images in a directory

    Args:
        image_dir: Directory containing test images
        recognizer: FaceRecognizer instance
        output_csv: Output CSV file path

    Returns:
        DataFrame with results
    """
    image_dir = Path(image_dir)

    # Find all images
    image_files = list(image_dir.glob('*.jpg')) + \
                  list(image_dir.glob('*.png')) + \
                  list(image_dir.glob('*.jpeg'))

    print(f"Found {len(image_files)} images to process")

    results = []

    for img_path in tqdm(image_files, desc="Processing images"):
        result = recognizer.recognize_face(str(img_path))

        if result['recognized']:
            person = result['person_info']
            results.append({
                'image_path': str(img_path),
                'recognized': True,
                'face_id': result['face_id'],
                'name': person['name'],
                'department': person['department'],
                'role': person['role'],
                'confidence': result['confidence']
            })
        else:
            results.append({
                'image_path': str(img_path),
                'recognized': False,
                'face_id': None,
                'name': None,
                'department': None,
                'role': None,
                'confidence': result['confidence']
            })

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Results saved to {output_csv}")

    # Print summary
    recognized_count = df['recognized'].sum()
    total_count = len(df)
    print(f"\nRecognition Summary:")
    print(f"  Total images: {total_count}")
    print(f"  Recognized: {recognized_count} ({recognized_count / total_count:.1%})")
    print(f"  Unknown: {total_count - recognized_count}")

    return df

# Example usage:
# results_df = batch_recognize_faces(
#     image_dir='/path/to/test/images',
#     recognizer=recognizer,
#     output_csv='recognition_results.csv'
# )