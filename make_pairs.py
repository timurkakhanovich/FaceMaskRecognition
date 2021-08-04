import numpy as np
import glob
import os

def makeMatches(file_path, file_spec, pairs_amount):
    persons = os.listdir(file_path)
    pairs_counter = 0

    while True:
        person = np.random.choice(persons)
        repeat_class = np.random.randint(2, 8)

        for _ in range(repeat_class):
            deuce = np.random.choice(os.listdir(os.path.join(file_path, person)), 2, replace=False)

            file_spec.write('{}\t{}\t{}\n'.format(person, deuce[0], deuce[1]))

            pairs_counter += 1

            if pairs_counter == pairs_amount:
                return


def makeMissmatches(file_path, file_spec, pairs_amount):
    persons = os.listdir(file_path)
    pairs_counter = 0

    while True:
        person1 = np.random.choice(persons)

        while True:
            person2 = np.random.choice(persons)

            if person1 != person2:
                break

        first = np.random.choice(os.listdir(os.path.join(file_path, person1)))
        second = np.random.choice(os.listdir(os.path.join(file_path, person2)))

        file_spec.write('{}\t{}\t{}\t{}\n'.format(person1, first, person2, second))

        pairs_counter += 1

        if pairs_counter == pairs_amount:
            return


def main():
    for _ in range(10):
        with open('lfw_pairs_test.txt', 'a') as f:
            makeMatches('Data/test', f, pairs_amount=200)
            makeMissmatches('Data/test', f, pairs_amount=200)

if __name__ == "__main__":
    main()
