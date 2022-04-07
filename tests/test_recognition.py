


from number_recognition import NumberRecognizer

n = NumberRecognizer()
n.load()
num = n.recognize('./numbers/5.png')
print(num)
