import csv

f = open('archive/Books.csv', 'r')
rdr = csv.reader(f)

for line in rdr:
    print(line)

f.close()