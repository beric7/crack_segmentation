import time
import logging


class TimeManager:
    def __init__(self):
        self.startTime = time.time()
            
    def show(self, msg='Time used'):
        usedTime = time.time() - self.startTime
        usedHour = int(usedTime // 3600)
        usedMin = int((usedTime - usedHour * 3600) // 60)
        usedSecond = int((usedTime - usedHour * 3600 - usedMin * 60))
        logging.info(f'{msg}: {usedHour} : {usedMin} : {usedSecond}')
        
    def restart():
        self.startTime = time.time()
    