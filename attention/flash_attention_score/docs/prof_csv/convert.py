import csv
import sys

# 方式一：从标准输入读取制表符分隔的数据
# 运行： python convert.py < input.tsv > output.csv
def main():
    reader = csv.reader(sys.stdin, delimiter='\t')
    writer = csv.writer(sys.stdout, delimiter=',', quoting=csv.QUOTE_ALL)
    for row in reader:
        writer.writerow(row)

if __name__ == '__main__':
    main()