# __author__ = ChiWun Yang
# __Email__ = yangcw5@mail2.sysu.edu.cn
# Copyright @SYSU AI
# This is a Project for "YiXianBei" Competition

import os
import argparse
from es_bot.interact import Inference


def cut(sentence):
    cut_set = {'。', '，', '】', '！', '；', '#', '~', '？'}
    n = len(sentence)
    for i in range(n):
        if sentence[n - i - 1] in cut_set:
            return sentence[:n - i - 1]
    return sentence


def set_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_file_path", type=str, default='./data.txt')
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--threshold_value", type=int, default=1)  # 筛选文本的阈值，按模型预测中的标签数来筛选
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--after_training", action="store_true")

    return parser


def main():
    parser = set_argparse()
    path = parser.parse_args().predict_file_path
    device = parser.parse_args().device
    batch_size = parser.parse_args().batch_size
    threshold_value = parser.parse_args().threshold_value
    max_len = parser.parse_args().max_len

    cmd = 'cd sentiment_analysis && ' + 'python predict.py --predict_file --predict_file_path .' + \
          path + ' --save_path ../outputs --batch_size ' + str(batch_size) + ' --device ' + device
    os.system(cmd)

    if parser.parse_args().after_training:
        model_path = './es_bot/model'
    else:
        model_path = './es_bot/min_ppl_model'

    bot = Inference(
        model_name_or_path=model_path,
        device=device,
        repetition_penalty=1.5,
        max_len=max_len
    )
    answers = ""
    with open('./outputs/sa_results.txt', encoding="utf-8") as file:
        for line in file.read().split('\n'):
            if list(line[:7]).count('1') >= threshold_value:
                answers += line[8:] + '\n'
                answers += cut(bot.predict(line[8:])) + '\n' + '\n'

    with open('./outputs/results.txt', encoding="utf-8", mode="w") as file:
        file.write(answers[:-2])


if __name__ == '__main__':
    main()
