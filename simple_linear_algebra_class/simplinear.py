class Matrix(object):

    def __init__(self, array):
        assert all([len(array[0]) == len(row) for row in array])
        self.array = array
        self.shape = (len(array), len(array[0]))

    def size(self):
        return self.shape

    def __getitem__(self, ids):
        if isinstance(ids, tuple):
            i, j = ids
            return self.array[i][j]
        else:
            return self.array[ids]

    def __setitem__(self, ids, item):
        i, j = ids
        self.array[i][j] = item

    def __str__(self):
        return '[' + '\n '.join([str(row) for row in self.array]) + ']'

    def __repr__(self):
        return 'Matrix:\n' + self.__str__()

    def __add__(self, other):
        res = []

        for x in range(self.shape[0]):
            if isinstance(other, Matrix):
                assert self.shape == other.shape
                row = [a + b for a, b in zip(self.array[x], other.array[x])]
            elif isinstance(other, (int, float)):
                row = [a + other for a in self.array[x]]
            res.append(row)

        return Matrix(res)

    def __sub__(self, other):
        res = []

        for x in range(self.shape[0]):
            if isinstance(other, Matrix):
                assert self.shape == other.shape
                row = [a - b for a, b in zip(self.array[x], other.array[x])]
            elif isinstance(other, (int, float)):
                row = [a - other for a in self.array[x]]
            res.append(row)

        return Matrix(res)

    def __mul__(self, other):
        res = []

        for x in range(self.shape[0]):
            if isinstance(other, Matrix):
                assert self.shape == other.shape
                row = [a * b for a, b in zip(self.array[x], other.array[x])]
            elif isinstance(other, (int, float)):
                row = [a * other for a in self.array[x]]
            res.append(row)

        return Matrix(res)

    def __matmul__(self, other):
        assert self.shape[1] == other.shape[0]
        m, n = other.shape[1], self.shape[0]
        other_t = other.transpose()
        res = []

        for i in range(n):
            row = []
            for j in range(m):
                item = sum([a * b for a, b in zip(self.array[i], other_t.array[j])])
                row.append(item)
            res.append(row)

        return Matrix(res)

    def dot(self, other):
        return self.__matmul__(other)

    def __truediv__(self, other):
        res = []

        for x in range(self.shape[0]):
            if isinstance(other, Matrix):
                assert self.shape == other.shape
                row = [a / b for a, b in zip(self.array[x], other.array[x])]
            elif isinstance(other, (int, float)):
                row = [a / other for a in self.array[x]]
            res.append(row)

        return Matrix(res)

    def transpose(self):
        return Matrix([list(tup) for tup in zip(*self.array)])

    def det(self):
        assert self.shape[0] == self.shape[1]
        trans = self.array.copy()
        m = self.shape[0]
        eps = 1e-9
        res = 1

        for i in range(m):
            if trans[i][i] == 0:
                trans[i][i] = eps
            for j in range(i + 1, m):
                factor = trans[j][i] / trans[i][i]
                for k in range(m):
                    trans[j][k] = trans[j][k] - factor * trans[i][k]

        for j in range(m):
            res *= trans[j][j]

        return res
