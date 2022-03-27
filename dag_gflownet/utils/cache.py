_PREV, _NEXT, _KEY, _VALUE = 0, 1, 2, 3

class LRUCache:
    def __init__(self, max_size):
        self.max_size = max_size
        self.mapping = {}

        # oldest
        self.head = [None, None, None, None]
        # newest
        self.tail = [self.head, None, None, None]
        self.head[_NEXT] = self.tail

    def __setitem__(self, key, value):
        link = self.mapping.get(key, self.head)

        if link is not self.head:
            raise KeyError(f'Key {key} already in cache.')

        if len(self.mapping) >= self.max_size:
            # Unlink the least recently used element
            _, old_next, old_key, _ = self.head[_NEXT]
            self.head[_NEXT] = old_next
            old_next[_PREV] = self.head
            del self.mapping[old_key]

        # Add new value as most recently used element
        last = self.tail[_PREV]
        link = [last, self.tail, key, value]
        self.mapping[key] = last[_NEXT] = self.tail[_PREV] = link

    def __getitem__(self, key):
        link = self.mapping.get(key, self.head)

        if link is self.head:
            raise KeyError(f'Key {key} not in cache.')

        # Unlink element from current position
        link_prev, link_next, key, value = link
        link_prev[_NEXT] = link_next
        link_next[_PREV] = link_prev

        # Add as most recently used element
        last = self.tail[_PREV]
        last[_NEXT] = self.tail[_PREV] = link
        link[_PREV] = last
        link[_NEXT] = self.tail

        return value

    def __contains__(self, key):
        return key in self.mapping

    def __len__(self):
        return len(self.mapping)

    def __str__(self):
        return str(dict(self.items()))

    def items(self):
        for value in self.mapping.values():
            yield (value[_KEY], value[_VALUE])
