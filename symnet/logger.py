import logging
import time


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

log_file = '{}_{}.log'.format('./output/train_log', time.strftime('%Y-%m-%d-%H-%M'))
fh = logging.FileHandler(log_file)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(fh)
logger.info('this is info message')
logger.warning('this is warn message')