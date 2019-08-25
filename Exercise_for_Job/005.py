def test( ):
    global a
    a = a+4

def b():
    global a

    for i in range(5):
        a = i
        test()
        print(a)
b()