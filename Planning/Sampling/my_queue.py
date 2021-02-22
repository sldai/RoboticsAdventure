import collections
import heapq
from heapq import *
import itertools

class QueueFIFO:
    """
    Class: QueueFIFO
    Description: QueueFIFO is designed for First-in-First-out rule.
    """

    def __init__(self):
        self.queue = collections.deque()

    def empty(self):
        return len(self.queue) == 0

    def put(self, node):
        self.queue.append(node)  # enter from back

    def get(self):
        return self.queue.popleft()  # leave from front


class QueueLIFO:
    """
    Class: QueueLIFO
    Description: QueueLIFO is designed for Last-in-First-out rule.
    """

    def __init__(self):
        self.queue = collections.deque()

    def empty(self):
        return len(self.queue) == 0

    def put(self, node):
        self.queue.append(node)  # enter from back

    def get(self):
        return self.queue.pop()  # leave from back


class QueuePrior:
    """
    Class: QueuePrior
    Description: QueuePrior reorders elements using value [priority]
    """

    def __init__(self):
        self.queue = []

    def empty(self):
        return len(self.queue) == 0

    def put(self, item, priority):
        heapq.heappush(self.queue, (priority, item))  # reorder s using priority

    def get(self):
        return heapq.heappop(self.queue)[1]  # pop out the smallest item

    def enumerate(self):
        return self.queue

class HeapDict:
    """A priority queue which can access the priority by key.
    Also it can update the priority of items in the heap.
    """
    def __init__(self):
        super().__init__()
        self.REMOVED = '<removed-task>' # placeholder for a removed task
        self.pq = []  # list of entries arranged in a heap
        self.entry_finder = {} # mapping of tasks to entries
        self.counter = itertools.count() # unique sequence count

    def add_task(self, task, priority=0):
        'Add a new task or update the priority of an existing task'
        if task in self.entry_finder:
            self.remove_task(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heappush(self.pq, entry)

    def remove_task(self, task):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = self.entry_finder.pop(task)
        entry[-1] = self.REMOVED

    def pop_task(self):
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while self.pq:
            priority, count, task = heappop(self.pq)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return task
        raise KeyError('pop from an empty priority queue')
    
    def __len__(self):
        return len(self.entry_finder)

    def empty(self):
        return len(self)==0

    def put(self, item, priority):
        """Wrapper for add task
        """
        self.add_task(item, priority)

    def get(self):
        """Wrapper for pop task
        """
        return self.pop_task()

    def remove(self, item):
        """Wrapper for remove task
        """
        self.remove_task(item)

    def top(self):
        """Peek the top item key and priority
        """
        while self.pq:
            if self.pq[0][-1] is not self.REMOVED:
                return self.pq[0][-1], self.pq[0][0]
            heappop(self.pq)
        raise KeyError('pop from an empty priority queue')

    def find(self, item):
        """Return True if find the item
        """
        return item in self.entry_finder.keys()