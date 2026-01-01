"""
Face Recognition Inference Engine
Matches query faces against database
"""

import numpy as np
import sqlite3
from typing import Optional, Dict, List, Tuple


class FaceRecognizer:
    """
    Face recognition inference engine
    Compares query face against database embeddings
    """

    def __init__(
            self,
            db_path: str,
            embedding_generator: object,
            threshold: float = 0.6
    ):
        """
        Args:
            db_path: Path to SQLite database
            embedding_generator: FaceEmbeddingGenerator instance
            threshold: Similarity threshold (0-1)
        """
        self.db_path = db_path
        self.embedding_generator = embedding_generator
        self.threshold = threshold

        # Load database embeddings
        self.db_embeddings = []  # List of (face_id, embedding)
        self._load_database()

        print(f"✓ Face recognizer initialized")
        print(f"  Database: {db_path}")
        print(f"  Threshold: {threshold}")
        print(f"  Registered faces: {len(self.db_embeddings)}")

    def _load_database(self):
        """Load all embeddings from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT face_id, embedding FROM embeddings')

        for face_id, embedding_bytes in cursor.fetchall():
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            self.db_embeddings.append((face_id, embedding))

        conn.close()
        print(f"✓ Loaded {len(self.db_embeddings)} embeddings")

    def cosine_similarity(
            self,
            embedding1: np.ndarray,
            embedding2: np.ndarray
    ) -> float:
        """Calculate cosine similarity (embeddings are L2-normalized)"""
        return float(np.dot(embedding1, embedding2))

    def recognize_face(self, image_path: str) -> Dict:
        """
        Recognize face in image

        Args:
            image_path: Path to query image

        Returns:
            Dictionary with recognition results
        """
        # Generate embedding for query
        query_embedding = self.embedding_generator.generate_embedding(image_path)

        if query_embedding is None:
            return {
                'recognized': False,
                'error': 'Failed to generate embedding',
                'face_id': None,
                'confidence': 0.0,
                'person_info': None
            }

        # Find best match
        best_match_id = None
        best_similarity = -1.0

        for face_id, db_embedding in self.db_embeddings:
            similarity = self.cosine_similarity(query_embedding, db_embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = face_id

        # Check threshold
        if best_similarity >= self.threshold:
            person_info = self._get_person_info(best_match_id)

            return {
                'recognized': True,
                'face_id': best_match_id,
                'confidence': float(best_similarity),
                'person_info': person_info
            }
        else:
            return {
                'recognized': False,
                'face_id': None,
                'confidence': float(best_similarity),
                'person_info': None
            }

    def _get_person_info(self, face_id: str) -> Optional[Dict]:
        """Retrieve person information from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT face_id, name, department, role, email, phone, notes
            FROM people WHERE face_id = ?
        ''', (face_id,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                'face_id': row[0],
                'name': row[1],
                'department': row[2],
                'role': row[3],
                'email': row[4],
                'phone': row[5],
                'notes': row[6]
            }
        return None

    def reload_database(self):
        """Reload embeddings from database"""
        self.db_embeddings = []
        self._load_database()