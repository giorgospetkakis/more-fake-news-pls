from os import listdir
import re

for directory in ["real/", "fake/"]:
    file_list = [f for f in listdir(directory)]
    for file in file_list:
        f = open(directory + file, "r")
        content = f.readlines()
        f.close()

        content = re.sub("\. ", "\n", "".join(content))
        f = open(directory + file, "w")
        content = f.writelines(content)
        f.close()
