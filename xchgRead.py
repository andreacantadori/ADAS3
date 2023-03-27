import sys

currentText = ''
exitLoop = False
while not exitLoop:
    xchgFile = open('xchgFile.txt','r')
    newText = xchgFile.read()
    xchgFile.close()
    if newText != currentText:
        currentText = newText
        print(currentText)
