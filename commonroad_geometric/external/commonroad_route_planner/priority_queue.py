import heapq
from typing import List


class PriorityQueue:
    """Class to represent a priority queue"""

    def __init__(self):
        self.elements = []
        # counter of elements in the queue
        self.count = 0

    def is_empty(self):
        """Returns true if the queue is empty"""

        return len(self.elements) == 0

    def push(self, item_id, item, priority):
        """Pushes an item into the queue and updates the counter

        :param item_id: the unique id of the element
        :param item: the element to be put in the queue
        :param priority: the value used to sort/prioritize elements in the queue. It's often the value of some cost function
        """
        heapq.heappush(self.elements, (priority, item_id, item))
        self.count += 1

    def pop(self):
        """Pops the item with the lowest priority."""
        if self.is_empty():
            return None

        best_element = None
        while (best_element is None) and (not self.is_empty()):
            best_element = heapq.heappop(self.elements)[2]

        return best_element

    def top(self):
        """Returns the item with the lowest priority."""
        if self.is_empty():
            return None

        return self.elements[0][2]

    def get_list(self, num_of_values: int = 1) -> List[type] or type:
        """Pops the items with the lowest priorities."""

        if self.is_empty():
            return None

        if num_of_values >= self.count:
            return_values = [element[2] for element in self.elements]
            return return_values

        return [heapq.heappop(self.elements)[2] for _ in range(num_of_values)]

    def get_item_ids(self) -> List[type] or type:
        """Returns a list of ids of the items in the queue."""

        return [element[1] for element in self.elements]

    def update_item_if_exists(self, updated_id, update_item, update_cost):
        """Updates an existing item in the queue.

        :param updated_id: id of item to be updated
        :param update_cost: cost of the item to be updated
        :param update_item: item to be updated
        """
        reverse_lookup = {item_id: index for index, (_, item_id, _) in enumerate(self.elements)}
        item_index = reverse_lookup.get(updated_id, -1)
        if item_index >= 0:
            (cost, i_id, item) = self.elements[item_index]
            if update_cost < cost:
                # make element invalid
                self.elements[item_index] = (cost, i_id, None)
                # put the new one in the queue
                self.push(updated_id, update_item, update_cost)

    def merge(self, other_queue: 'PriorityQueue'):
        """Merges with another priority queue."""

        if other_queue is None or other_queue.is_empty():
            return

        if self.is_empty():
            self.elements = other_queue.elements
            self.count = other_queue.count
            return

        self.elements = list(heapq.merge(self.elements, other_queue.elements, key=lambda c: c[0]))
        self.count += other_queue.count

    def __str__(self):
        return f"{self.elements}"
