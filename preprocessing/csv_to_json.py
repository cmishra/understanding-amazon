import pandas
import os
import argparse
import json


def csv_to_json(csv_path, output_path):
    targets = pandas.read_csv(csv_path)
    tag_list = set()
    for tag_description in targets['tags']:
        tags = tag_description.split(' ')
        for t in tags:
            tag_list.add(t)
    for t in tag_list:
        targets[t] = targets['tags'].apply(lambda x: 1 if t in x.split(' ') else 0)

    target_mappings = {}
    image_names = targets["image_name"]
    for i in xrange(targets.shape[0]):
        row = targets.iloc[i]
        target_mappings[row[0]] = row[2:].tolist()

    with open(output_path, 'w') as f:
        json.dump(target_mappings, f)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Processes raw target csv into json")
    parser.add_argument("--train_csv",
                        help="path to extracted csv file")
    parser.add_argument("--output",
                        help="where to save json file")
    args = parser.parse_args()
    
    csv_to_json(args.train_csv, args.output)
    
    

