import threading
def thread_job():
    print("this is added thread ,number is %s"%threading.current_thread())
    print(threading.active_count())
def main():
    add_thread = threading.Thread(target=thread_job)
    add_thread.start()
    add_thread1 = threading.Thread(target=thread_job)
    add_thread1.start()
    print(threading.active_count())
    print(threading.enumerate())
    print(threading.current_thread())
if __name__ == '__main__':
    main()