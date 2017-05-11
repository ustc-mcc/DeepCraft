import json
import os
import sys
import glob

json_file = open('vis.json')
for line in json_file:
#    line = line.strip('[').strip(']')
    json_obj = json.loads(line)
    for i in range(len(json_obj)):
        print(json_obj[i]['caption'])
    
#    print(a)

