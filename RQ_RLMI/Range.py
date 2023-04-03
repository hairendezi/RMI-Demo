class Range:
    def __init__(self, rangeConfig):
        self.LOW = rangeConfig["LOW"]
        self.HIGH = rangeConfig["HIGH"]
        self.ID = rangeConfig["ID"]

    def __str__(self):
        return "Range %d: [%d, %d)" % (self.ID, self.LOW, self.HIGH+1)

    def matchPos(self, pos) -> bool:
        if self.LOW <= pos <= self.HIGH:
            return True
        else:
            return False
