class KVClass:
    def __init__(self, key, value, rangeID: []):
        self.key = key
        self.value = value
        self.rangeID = rangeID

    def __str__(self):
        return "key: %d, value: %.10f, rangeID: %s" % (self.key, self.value, ", ".join([str(t) for t in self.rangeID]))