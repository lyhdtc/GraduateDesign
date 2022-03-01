import time

def timmer(func):
    print('run timmer')
    def deco(*args, **kwargs):
        print('\n函数： {_funcname_} 开始运行：'.format(_funcname_ = func.__name__))
        
        # time.time()方法调用的是系统时间，这里改用更准确的cpu时间测试
        # start_time = time.time()
        start_time = time.perf_counter()
        res = func(*args, **kwargs)
        end_time = time.perf_counter()
        print('函数:{_funcname_}运行了 {_time_}秒'
              .format(_funcname_=func.__name__, _time_=(end_time - start_time)))
        return res
    return deco