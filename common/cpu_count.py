import multiprocessing
import sys

def get_dataloader_workers():  #@save
    # macos
    if sys.platform.startswith('darwin'):
        # 获取最大可用的进程数
        max_workers = multiprocessing.cpu_count()
        # print(f'macos max_workers: {max_workers}')
        return 0
    # windows
    elif sys.platform.startswith('win'):
   
        max_workers = multiprocessing.cpu_count()
        # print(f'windows max_workers: {max_workers}')
        return 0
    # linux
    else:
        max_workers = multiprocessing.cpu_count()
        # print(f'linux max_workers: {max_workers}')
        return max_workers - 1


def main():
    return get_dataloader_workers()

if __name__ == '__main__':
    main()
