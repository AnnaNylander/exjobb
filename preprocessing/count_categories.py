import os

PATH = '/media/annaochjacob/crucial/dataset/eukaryote/train/'

def main():

    counts = {'left': 0,
              'right': 0,
              'left_intention': 0,
              'right_intention': 0,
              'straight': 0,
              'traffic_light': 0,
              'other': 0,}

    for fruit in os.listdir(PATH):
        print(fruit)
        for category in os.listdir(PATH + fruit):
            outputs = os.listdir(PATH + fruit + '/' + category + '/output/')
            n_outputs = len(outputs)
            counts[category] += n_outputs
            print('\t', n_outputs, '\t', category)

    print('Total:')
    for key, val in counts.items():
        print(val, '\t', key)


if __name__ == "__main__":
    main()
