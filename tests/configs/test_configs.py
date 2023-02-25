from cv.utils import Config


if __name__ == '__main__':
    cfg = Config.fromfile('./config_file.py')
    print(cfg.pretty_text)