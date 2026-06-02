import csv
import random
import numpy as np
import warnings

warnings.filterwarnings("ignore")

SEED = 1338
random.seed(SEED)
np.random.seed(SEED)


def read_csv(file_path):
    """Read a CSV file and return rows as a list."""
    data = []
    with open(file_path, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)
    return data


def save_csv(data, file_path):
    """Save data to a CSV file."""
    with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)


def get_unique_items(data, column_index):
    """Extract unique items from a specified column while preserving order."""
    unique_items = []
    seen = set()

    for row in data:
        item = row[column_index]
        if item not in seen:
            unique_items.append(item)
            seen.add(item)

    return unique_items


def generate_negative_samples(positive_samples, all_circrnas, all_diseases):
    """
    Randomly generate negative circRNA-disease pairs.

    Negative samples are selected from all possible circRNA-disease combinations
    that do not appear in the positive sample set.
    """
    positive_set = set(tuple(pair) for pair in positive_samples)
    negative_set = set()
    negative_samples = []

    while len(negative_samples) < len(positive_samples):
        circrna = random.choice(all_circrnas)
        disease = random.choice(all_diseases)
        candidate_pair = (circrna, disease)

        if candidate_pair in positive_set:
            continue

        if candidate_pair in negative_set:
            continue

        negative_set.add(candidate_pair)
        negative_samples.append([circrna, disease])

    return negative_samples


def main():
    interaction_file = "dataset/circAtlas 3.0/interaction.csv"

    negative_sample_file = "NegativeSample.csv"
    all_circrna_file = "AllCircRNA.csv"
    final_sample_file = "Positive_Negative_Samples.csv"

    original_data = read_csv(interaction_file)
    print("Number of original interactions:", len(original_data))

    positive_samples = [[row[0], row[1]] for row in original_data]
    print("Number of positive samples:", len(positive_samples))

    all_diseases = get_unique_items(original_data, column_index=1)
    all_circrnas = get_unique_items(original_data, column_index=0)

    print("Number of diseases:", len(all_diseases))
    print("Number of circRNAs:", len(all_circrnas))

    save_csv([[item] for item in all_circrnas], all_circrna_file)

    negative_samples = generate_negative_samples(
        positive_samples=positive_samples,
        all_circrnas=all_circrnas,
        all_diseases=all_diseases
    )

    print("Number of negative samples:", len(negative_samples))
    save_csv(negative_samples, negative_sample_file)

    final_samples = np.vstack((positive_samples, negative_samples))
    print("Final sample shape:", final_samples.shape)

    save_csv(final_samples, final_sample_file)


if __name__ == "__main__":
    main()