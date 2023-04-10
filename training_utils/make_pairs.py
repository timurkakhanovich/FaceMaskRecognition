from pathlib import Path
import numpy as np
from tqdm import tqdm


def make_matches(
    data_path: Path,
    target_file: Path,
    pairs_amount: int,
) -> None:
    persons = list(data_path.glob("*"))

    for _ in tqdm(range(pairs_amount), desc="Creating matches"):
        person = np.random.choice(persons)
        repeat_class = np.random.randint(2, 8)

        for _ in range(repeat_class):
            person_dir = list(person.glob("**/*"))
            deuce = np.random.choice(person_dir, 2, replace=False)

            with target_file.open("a") as fin:
                fin.write(f"{person}\t{deuce[0]}\t{deuce[1]}\n")


def make_missmatches(
    data_path: Path,
    target_file: Path,
    pairs_amount: int,
):
    persons = list(data_path.glob("*"))

    for _ in tqdm(range(pairs_amount), desc="Creating mismatches"):
        person1 = np.random.choice(persons)

        while True:
            person2 = np.random.choice(persons)

            if person1 != person2:
                break

        first = np.random.choice(list(person1.glob("**/*")))
        second = np.random.choice(list(person2.glob("**/*")))

        with target_file.open("a") as fin:
            fin.write(f"{person1}\t{first}\t{person2}\t{second}\n")


def main() -> None:
    source_dir = Path("Data/test")
    target_file = Path("lfw_pairs_test.txt")

    make_matches(source_dir, target_file, pairs_amount=200)
    make_missmatches(source_dir, target_file, pairs_amount=200)
