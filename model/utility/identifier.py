class Allocator:
    def __init__(self):
        self.next_id = 0
        return

    def get_next_id(self):
        id = self.next_id
        self.next_id += 1
        return id

