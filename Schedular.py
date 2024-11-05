class Process:
    def __init__(self, pid, priority, arrival_time, burst_time):
        self.pid = pid
        self.priority = priority
        self.arrival_time = arrival_time
        self.burst_time = burst_time
        self.remaining_time = burst_time

class Scheduler:
    def __init__(self):
        self.queue1 = []  # Highest priority queue
        self.queue2 = []  # Medium priority queue
        self.queue3 = []  # Lowest priority queue

    def add_process(self, process):
        if process.priority == 1:
            self.queue1.append(process)
        elif process.priority == 2:
            self.queue2.append(process)
        else:
            self.queue3.append(process)

    def execute(self):
        while self.queue1 or self.queue2 or self.queue3:
            if self.queue1:
                process = self.queue1.pop(0)
                nb = 1
            elif self.queue2:
                process = self.queue2.pop(0)
                nb = 2
            else:
                process = self.queue3.pop(0)
                nb = 3

            print(f"Executing process {process.pid} in queue {nb}")

            # Simulate execution of process
            if process.remaining_time > 2:
                process.remaining_time -= 2
                # Demote process to lower priority queue
                if process.priority < 3:
                    process.priority += 1
                self.add_process(process)
            else:
                print(f"Process {process.pid} completed.")

if __name__ == "__main__":
    # Create processes
    processes = [
        Process(1, 1, 0, 5),
        Process(2, 2, 1, 4),
        Process(3, 3, 2, 3),
        Process(4, 1, 3, 2),
        Process(5, 2, 4, 1)
    ]

    # Create scheduler
    scheduler = Scheduler()

    # Add processes to scheduler
    for process in processes:
        scheduler.add_process(process)

    # Execute processes
    scheduler.execute()



































