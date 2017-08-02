#!/opt/share/Python-3.5.2/x86_64/bin/python3

import argparse
import os

def move_file(source, dest, f):
    os.rename(
        os.path.join(source, f),
        os.path.join(dest, f),
    )
              

def split(training_proportion, 
          val_proportion,
          source_dir,
          destination_dir):

    validation_folder_name = 'val'
    test_folder_name = 'test'
    train_folder_name = 'train'

    if not os.path.isdir(source_dir):
        exit("source directory doesn't exist.")

    if not os.path.isdir(destination_dir):
        os.mkdir(destination_dir)
        exit("Destination directory doesn't exist. Run the makefile.")
    else:
        print("Destination directory exists.")
        files = os.listdir(destination_dir)
        for f in [validation_folder_name, 
                  test_folder_name,
                  train_folder_name]:
            if f not in files:
                os.mkdir(os.path.join(destination_dir, f))
                print("Folder doesn't exist. Creating %s." % f)
            else:
                print("Using existing folder %s." % f)

    images = os.listdir(source_dir)
    N = len(images)
    training_size = round(N*training_proportion)
    validation_size = round(N*val_proportion)
    test_size = N - training_size - validation_size
    
    print("Training images %d" % training_size)
    print("Validation images %d" % validation_size)
    print("Test images %d" % test_size)

    for i in range(N):
        if i < training_size:
            move_file(source_dir, 
                      os.path.join(destination_dir, 
                                   train_folder_name),
                      images[i])
        elif i < (training_size + validation_size):
            move_file(source_dir, 
                      os.path.join(destination_dir,
                                   validation_folder_name),
                      images[i])
        else:
            move_file(source_dir, 
                      os.path.join(destination_dir,
                                   test_folder_name),
                      images[i])
                                   
    print("Images split!")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Splits dataset into multiple datasets for training")
    parser.add_argument('--train_proportion', 
                        type=float, 
                        help="proportion of data to use for training. 0<x<1")
    parser.add_argument('--val_proportion',
                        type=float,
                        help="proportion of data to use for validation. 0<x<1")
    parser.add_argument('--source_dir',
                        help="directory with full training data")
    parser.add_argument("--destination_dir",
                        help="directory to put data splits in")

    args = parser.parse_args()

    split(args.train_proportion,
          args.val_proportion,
          args.source_dir,
          args.destination_dir)

    

