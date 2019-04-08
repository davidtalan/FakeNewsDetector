import re
import chardet

with open('/home/david/2019-ca400-taland2/src/dataset/concat.csv', 'rb') as f:
     result = chardet.detect(f.readline())
     print (result['encoding'])

#print (result['encoding'])
