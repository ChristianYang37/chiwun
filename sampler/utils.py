def save_texts(input_texts, path='./outputs/data.txt'):
    s = ''
    for texts in input_texts:
        for text in texts:
            s += text + '\n'
        s += '\n'
    with open(path, encoding='utf-8', mode='w') as file:
        file.write(s[:-2])


def read_texts(path):
    ret = []
    with open(path, encoding='utf-8', mode='r') as file:
        data = file.read().split('\n\n')
    for texts in data:
        ret.append(texts.split('\n'))
    return ret
