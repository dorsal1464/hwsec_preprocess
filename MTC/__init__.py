import threading
import numpy as np


class WorkerThread(threading.Thread):
    def work(self, data):
        dim_count = np.ndim(data)
        for i in range(0, np.shape(data)[self.workdim]):
            self.target(data[i])

    def __init__(self, target, data, workdim):
        self.target = target
        self.workdim = workdim
        super().__init__(target=self.work, args={data})
        self.progress = 0





class MultiThreadComputation:
    def __init__(self, n_threads: int, target, debug: bool):
        self.nThreads = n_threads
        self.target = target
        self.threadPool = []
        self.debug = debug
        self.debugThread = None

    def debugFunc(self):
        pass

    # inputs: multi-dimensional data, dimension to split the data
    def start(self, data, dim, workdim):
        sz = np.shape(data)[dim]
        for i in range(0, self.nThreads):
            t = WorkerThread(self.target, data[i*sz, i*sz+1], workdim)
            t.start()
            self.threadPool.append(t)
        if self.debug:
            t = threading.Thread(target=self.debugFunc)
            t.start()
            self.debugThread = t
