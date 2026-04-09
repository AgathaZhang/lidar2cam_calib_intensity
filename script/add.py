import os
import glob
from rosbag import Bag


import os

def merge_bag_files(folder):
    merged_bag_file = os.path.join(folder, 'merged.bag')
    if os.path.exists(merged_bag_file):
        os.remove(merged_bag_file)
    bag_files = glob.glob(os.path.join(folder, '*.bag'))
    bag_files.sort()

    with Bag(merged_bag_file, 'w') as merged_bag:
        for bag_file in bag_files:
            with Bag(bag_file, 'r') as bag:
                for topic, msg, t in bag.read_messages(): 
                    merged_bag.write(topic, msg, t)

if __name__ == '__main__':
    merge_bag_files("/home/kilox/workspace/calibr_data/11111/123")