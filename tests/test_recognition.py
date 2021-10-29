


from number_recognition import NumberRecogniser

n = NumberRecogniser()
n.load()
num = n.recognize('./numbers/5.png')
print(num)
