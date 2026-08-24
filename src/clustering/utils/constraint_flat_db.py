import lmdb
import struct

import numpy as np


class ConstraintFlatDB:
    def __init__(self, db_path='./constraint_flat_db'):
        self.env = lmdb.open(db_path, map_size=100 * 1024 ** 3)

    def _make_key(self, dict_key, set_member):
        """Convert (dict_key, set_member) to bytes"""
        return struct.pack('2i', dict_key, set_member)

    def add_tuple(self, tuple_pair):
        """Needs write=True for modifications"""
        dict_key, set_member = tuple_pair
        with self.env.begin(write=True) as txn:
            txn.put(self._make_key(dict_key, set_member), b'')

    def add_tuples(self, tuple_list):
        """Add multiple tuples in batch"""
        with self.env.begin(write=True) as txn:
            for dict_key, set_member in tuple_list:
                txn.put(self._make_key(dict_key, set_member), b'')

    def contains_tuple(self, tuple_pair):
        """Read-only - no write=True needed"""
        dict_key, set_member = tuple_pair
        with self.env.begin() as txn:  # Faster, allows concurrency
            return txn.get(self._make_key(dict_key, set_member)) is not None

    def read_all_tuples(self):
        """Memory-efficient iterator over all tuples"""
        with self.env.begin() as txn:
            cursor = txn.cursor()
            cursor.first()
            while True:
                key = cursor.key()
                if len(key) == 8:
                    dict_key, set_member = struct.unpack('2i', key)
                    yield (dict_key, set_member)
                if not cursor.next():
                    break

    def get_all_tuples_as_list(self):
        """Read the full database and return as List[Tuple[int, int]]"""
        tuples_list = []
        with self.env.begin() as txn:
            cursor = txn.cursor()
            cursor.first()
            while True:
                key = cursor.key()
                if len(key) == 8:
                    dict_key, set_member = struct.unpack('2i', key)
                    tuples_list.append((dict_key, set_member))
                if not cursor.next():
                    break
        return tuples_list

    def get_all_tuples_as_numpy(self) -> np.ndarray:
        """Return all constraints as a numpy array of shape (N, 2), dtype int32.

        Pre-allocates the array using the DB entry count to avoid building
        a Python list of tuples, which would be ~14x larger in memory.
        """
        with self.env.begin() as txn:
            n = txn.stat()['entries']
        arr = np.empty((n, 2), dtype=np.int32)
        idx = 0
        with self.env.begin() as txn:
            cursor = txn.cursor()
            if cursor.first():
                while True:
                    key = cursor.key()
                    if len(key) == 8:
                        arr[idx, 0], arr[idx, 1] = struct.unpack('2i', key)
                        idx += 1
                    if not cursor.next():
                        break
        return arr[:idx]

    def close(self):
        self.env.close()