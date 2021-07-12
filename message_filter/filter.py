import re


class SpamFilter(object):

    def __init__(self, inp_file):
        self.inp_file = inp_file

    def word_tokenize(self, line):
        return line.split()

    def word_punct_tokenize(self, line):
        return re.compile(r'\w+|[^\w\s]+').findall(line)

    def is_not_spam(self, line, tokenize_func):
        tokens = tokenize_func(line)
        symbols = (s for t in tokens for s in t if not t.isdigit())
        for s in symbols:
            if s.isdigit():
                break
        else:
            return True

    def get_filtered(self, consider_punct=True):
        if consider_punct:
            tokenize_func = self.word_punct_tokenize
        else:
            tokenize_func = self.word_tokenize

        with open(self.inp_file, 'r') as inp:
            with open('./filtered_messages.txt', 'w') as out:
                for line in inp:
                    if self.is_not_spam(line, tokenize_func):
                        out.write(line)
        print('Done!')


if __name__ == '__main__':
    inp_file = input('Enter path: ')
    sf = SpamFilter(inp_file)
    sf.get_filtered()
