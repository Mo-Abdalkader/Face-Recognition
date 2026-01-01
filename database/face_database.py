"""
Thread-Safe SQLite Database Interface for Face Recognition
Fixes: SQLite objects created in a thread can only be used in that same thread
"""

import sqlite3
import numpy as np
from typing import Optional, List, Dict, Tuple
from datetime import datetime
import threading


class FaceDatabase:
    """
    Thread-safe SQLite database manager for face recognition system
    Each thread gets its own connection
    """

    def __init__(self, db_path: str = 'face_recognition.db'):
        """
        Initialize database with thread-safe connections

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._local = threading.local()

        # Create tables in main thread
        self._create_tables()
        print(f"✓ Connected to database: {self.db_path}")

    def _get_connection(self):
        """Get thread-local database connection"""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,  # Allow use across threads
                timeout=30.0  # Increased timeout
            )
            # Enable foreign keys
            self._local.conn.execute("PRAGMA foreign_keys = ON")
        return self._local.conn

    def _get_cursor(self):
        """Get cursor for current thread"""
        return self._get_connection().cursor()

    def _create_tables(self):
        """Create database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Embeddings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                image_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (face_id) REFERENCES people(face_id)
            )
        ''')

        # People metadata table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS people (
                face_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                department TEXT,
                role TEXT,
                email TEXT,
                phone TEXT,
                notes TEXT,
                registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Recognition history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recognition_history (
                history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                query_image_path TEXT,
                result_type TEXT,
                matched_face_id TEXT,
                matched_name TEXT,
                confidence REAL,
                threshold_used REAL,
                FOREIGN KEY (matched_face_id) REFERENCES people(face_id)
            )
        ''')

        # Indices for faster lookups
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_face_id 
            ON embeddings(face_id)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_history_timestamp 
            ON recognition_history(timestamp)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_history_result_type 
            ON recognition_history(result_type)
        ''')

        conn.commit()
        conn.close()
        print("✓ Database tables created/verified")

    def add_person(
        self,
        face_id: str,
        name: str,
        department: Optional[str] = None,
        role: Optional[str] = None,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        notes: Optional[str] = None
    ) -> bool:
        """Add new person to database (thread-safe)"""
        try:
            cursor = self._get_cursor()
            cursor.execute('''
                INSERT INTO people (face_id, name, department, role, email, phone, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (face_id, name, department, role, email, phone, notes))
            self._get_connection().commit()
            print(f"✓ Added person: {name} (ID: {face_id})")
            return True
        except sqlite3.IntegrityError:
            print(f"✗ Person with face_id '{face_id}' already exists!")
            return False
        except Exception as e:
            print(f"✗ Error adding person: {e}")
            return False

    def add_embedding(
        self,
        face_id: str,
        embedding: np.ndarray,
        image_path: Optional[str] = None
    ) -> Optional[int]:
        """Store face embedding (thread-safe)"""
        try:
            # Verify face_id exists
            cursor = self._get_cursor()
            cursor.execute('SELECT face_id FROM people WHERE face_id = ?', (face_id,))
            if not cursor.fetchone():
                print(f"✗ face_id '{face_id}' not found. Add person first!")
                return None

            # Serialize embedding
            embedding_bytes = embedding.astype(np.float32).tobytes()

            cursor.execute('''
                INSERT INTO embeddings (face_id, embedding, image_path)
                VALUES (?, ?, ?)
            ''', (face_id, embedding_bytes, image_path))
            self._get_connection().commit()

            embedding_id = cursor.lastrowid
            print(f"✓ Added embedding {embedding_id} for face_id: {face_id}")
            return embedding_id

        except Exception as e:
            print(f"✗ Error adding embedding: {e}")
            return None

    def get_all_embeddings(self) -> List[Tuple[str, np.ndarray, str]]:
        """Retrieve all embeddings (thread-safe)"""
        cursor = self._get_cursor()
        cursor.execute('''
            SELECT face_id, embedding, image_path 
            FROM embeddings
        ''')

        results = []
        for face_id, embedding_bytes, image_path in cursor.fetchall():
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            results.append((face_id, embedding, image_path))

        return results

    def get_embeddings_by_face_id(self, face_id: str) -> List[np.ndarray]:
        """Get all embeddings for specific person (thread-safe)"""
        cursor = self._get_cursor()
        cursor.execute('''
            SELECT embedding FROM embeddings WHERE face_id = ?
        ''', (face_id,))

        embeddings = []
        for (embedding_bytes,) in cursor.fetchall():
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            embeddings.append(embedding)

        return embeddings

    def get_person_info(self, face_id: str) -> Optional[Dict]:
        """Retrieve person metadata (thread-safe)"""
        cursor = self._get_cursor()
        cursor.execute('''
            SELECT face_id, name, department, role, email, phone, notes, registered_at
            FROM people WHERE face_id = ?
        ''', (face_id,))

        row = cursor.fetchone()
        if row:
            return {
                'face_id': row[0],
                'name': row[1],
                'department': row[2],
                'role': row[3],
                'email': row[4],
                'phone': row[5],
                'notes': row[6],
                'registered_at': row[7]
            }
        return None

    def get_all_face_ids(self) -> List[str]:
        """Get list of all registered face IDs (thread-safe)"""
        cursor = self._get_cursor()
        cursor.execute('SELECT face_id FROM people')
        return [row[0] for row in cursor.fetchall()]

    def delete_person(self, face_id: str) -> bool:
        """Delete person and all embeddings (thread-safe)"""
        try:
            cursor = self._get_cursor()
            cursor.execute('DELETE FROM embeddings WHERE face_id = ?', (face_id,))
            cursor.execute('DELETE FROM people WHERE face_id = ?', (face_id,))
            self._get_connection().commit()
            print(f"✓ Deleted person and embeddings for face_id: {face_id}")
            return True
        except Exception as e:
            print(f"✗ Error deleting person: {e}")
            return False

    def get_stats(self) -> Dict:
        """Get database statistics (thread-safe)"""
        cursor = self._get_cursor()

        cursor.execute('SELECT COUNT(*) FROM people')
        num_people = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM embeddings')
        num_embeddings = cursor.fetchone()[0]

        return {
            'total_people': num_people,
            'total_embeddings': num_embeddings,
            'avg_embeddings_per_person': num_embeddings / num_people if num_people > 0 else 0
        }

    def close(self):
        """Close database connection for current thread"""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
            print("✓ Database connection closed")