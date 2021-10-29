


from number_recognition import NumberRecogniser

n = NumberRecogniser()
n.load()
num = n.recognize('/home/mauro/work/number-recognition/tests/numbers/5.png')
print(num)
