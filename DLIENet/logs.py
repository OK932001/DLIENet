import os
import time
import torch
import datetime


def Write_time(time=None, mode='start', root=None, filename=None):
    with open(os.path.join(root, filename), mode="a", encoding='utf-8') as f:

        # 结束才会输出这个信息
        if mode == 'end':
            # 持续时长
            duration = str(datetime.timedelta(seconds=int(time)))
            f.write(mode + ' : ' + duration)

        f.write('\ntrain {} time: {}\n'.format(
            mode,
            datetime.datetime.now().strftime("%Y-%D-%D %H:%M:%S")))


# 向文件中记录信息
def write_infor(root, filename='result', category=None, infor=None):
    # 保存文件的路径
    file_path = os.path.join(root, filename)
    with open(file_path, mode='a', encoding='utf-8') as f:
        # 记录开始时刻
        if category == 'start_time' or category == 'end_time':
            f.write('{} : {}\n'.format(
                category,
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        # 记录训练时长
        elif category == 'duration':
            duration = str(datetime.timedelta(seconds=int(infor)))
            f.write(category + ' : ' + duration + '\n')
        # 记录其他信息
        else:
            f.write(category.ljust(20) + ':' + str(infor) + '\n')


if __name__ == '__main__':
    if not os.path.exists('./result/A25'):
        os.makedirs('./result/A25')
    root = './result/A25'
    filename = 'result_log.txt'
    file_path = os.path.join(root, filename)

    # Write_time(time.time(), mode='start', root=root, filename=filename)
    start_time = time.time()
    write_infor(root, filename, 'start_time', start_time)
    write_infor(root, filename, 'Epoch', 5)
    write_infor(root, filename, 'train_loss', 0.05)
    a = torch.tensor([1, 2])
    write_infor(root, filename, 'mean', a)
    end_time = time.time()
    write_infor(root, filename, 'end_time', end_time)
    write_infor(root, filename, 'duration', end_time - start_time)
