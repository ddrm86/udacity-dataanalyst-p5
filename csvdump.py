import pickle
import csv

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

with open('enron.csv', 'w') as csvfile:
    fieldnames = data_dict.values()[0].keys()
    fieldnames.append('name')
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for name in data_dict.keys():
        data_dict[name]['name'] = name
        writer.writerow(data_dict[name])
