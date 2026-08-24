import lmdb
import struct


class ConstraintGraphDB:
    def __init__(self, db_path='./constraint_graph_db'):
        self.env = lmdb.open(db_path, map_size=100 * 1024 ** 3)

    def _key_to_bytes(self, dict_key):
        """Convert dict_key to bytes"""
        return struct.pack('i', dict_key)

    def _serialize_set(self, members):
        """Pack set of integers to bytes"""
        if not members:
            return struct.pack('I', 0)

        sorted_members = sorted(members)
        return struct.pack(f'I{len(sorted_members)}i', len(sorted_members), *sorted_members)

    def _deserialize_set(self, data):
        """Unpack bytes back to set of integers"""
        if not data:
            return set()

        count = struct.unpack('I', data[:4])[0]
        if count == 0:
            return set()

        integers = struct.unpack(f'{count}i', data[4:])
        return set(integers)

    def add_set(self, dict_key, members):
        """Store entire set for dict_key"""
        with self.env.begin(write=True) as txn:
            txn.put(self._key_to_bytes(dict_key), self._serialize_set(members))

    def add_sets(self, dict_of_sets):
        """Add multiple sets from Dict[int, Set[int]]"""
        with self.env.begin(write=True) as txn:
            for dict_key, members in dict_of_sets.items():
                txn.put(self._key_to_bytes(dict_key), self._serialize_set(members))

    def add_to_set(self, dict_key, member):
        """Add single member to set"""
        members = self.get_set(dict_key)
        members.add(member)
        self.add_set(dict_key, members)

    def update_sets(self, dict_of_sets):
        """Merge sets - add if doesn't exist, merge if exists"""
        with self.env.begin(write=True) as txn:
            for dict_key, new_members in dict_of_sets.items():
                existing_data = txn.get(self._key_to_bytes(dict_key))
                existing_set = self._deserialize_set(existing_data) if existing_data else set()
                merged_set = existing_set | set(new_members)  # Union
                txn.put(self._key_to_bytes(dict_key), self._serialize_set(merged_set))

    def contains(self, dict_key, member):
        """Check if member is in set for dict_key"""
        members = self.get_set(dict_key)
        return member in members

    def get_set(self, dict_key):
        """Get set for dict_key (returns empty set if doesn't exist)"""
        with self.env.begin() as txn:
            data = txn.get(self._key_to_bytes(dict_key))
            return self._deserialize_set(data) if data else set()

    def read_all_sets(self):
        """Memory-efficient iterator over all (dict_key, set) pairs"""
        with self.env.begin() as txn:
            cursor = txn.cursor()
            cursor.first()
            while True:
                key_bytes = cursor.key()
                value_bytes = cursor.value()

                dict_key = struct.unpack('i', key_bytes)[0]
                members = self._deserialize_set(value_bytes)

                yield (dict_key, members)

                if not cursor.next():
                    break

    def get_all_sets_as_dict(self):
        """Read the full database and return as Dict[int, Set[int]]"""
        constraint_dict = {}
        with self.env.begin() as txn:
            cursor = txn.cursor()
            cursor.first()
            while True:
                key_bytes = cursor.key()
                value_bytes = cursor.value()

                dict_key = struct.unpack('i', key_bytes)[0]
                members = self._deserialize_set(value_bytes)
                constraint_dict[dict_key] = members

                if not cursor.next():
                    break
        return constraint_dict

    def close(self):
        self.env.close()