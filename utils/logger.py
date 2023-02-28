import datetime


def log(msg, savePath=None, timeStamp=False, printDefault=True):
    if timeStamp:
        time = datetime.datetime.now()
        tem = '%s: %s' % (time, msg)
    else:
        tem = '%s' % msg
    if savePath is not None:
        with open(savePath, 'a') as f:
            f.write(tem + '\n')
    if printDefault:
        print(tem)

