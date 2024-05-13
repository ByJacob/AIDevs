from base import BaseTasks


class Task(BaseTasks):
    def __init__(self):
        super().__init__("helloapi")

    def resolve(self, task):
        return task['cookie']


if __name__ == "__main__":
    t = Task()
    t.process()
