from matplotlib import pyplot as plt
import csv
 
# opening the CSV file
with open('output1.csv', mode ='r') as f1: 
  # reading the CSV file
  output1 = list(csv.reader(f1))

# opening the CSV file
with open('output2.csv', mode ='r') as f2: 
  # reading the CSV file
  output2 = list(csv.reader(f2))

for m1, m2 in zip(output1, output2):
    plt.scatter(m1, m2)
    plt.xlabel("Function 1")
    plt.ylabel("Function 2")
    plt.title("Pareto front")
    plt.show()
 
