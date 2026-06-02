import csv


def load_node_embeddings(file_path):
    """
    Load node embeddings from a CSV file.

    Format:
    node_id, feature_1, feature_2, ...
    """
    embedding_dict = {}

    with open(file_path, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)

        for row in reader:
            node_id = row[0]
            embedding = row[1:]
            embedding_dict[node_id] = embedding

    return embedding_dict


def generate_sample_features(pair_file, embedding_dict, output_file):
    """
    Replace node IDs in sample pairs with their corresponding embeddings,
    and concatenate the embeddings of the two nodes.
    """
    with open(pair_file, "r", encoding="utf-8") as infile:
        with open(output_file, "w", newline="", encoding="utf-8") as outfile:
            writer = csv.writer(outfile)

            for row in csv.reader(infile):
                node_1, node_2 = row

                embedding_1 = embedding_dict.get(node_1, [])
                embedding_2 = embedding_dict.get(node_2, [])

                combined_embedding = embedding_1 + embedding_2
                writer.writerow(combined_embedding)


def main():
    embedding_file = "node_embeddings.csv"
    sample_pair_file = "Positive_Negative_Samples.csv"
    output_feature_file = "SampleFeatures.csv"

    embedding_dict = load_node_embeddings(embedding_file)

    print("Number of loaded node embeddings:", len(embedding_dict))

    generate_sample_features(
        sample_pair_file,
        embedding_dict,
        output_feature_file
    )

    print(f"Sample feature matrix has been saved to: {output_feature_file}")


if __name__ == "__main__":
    main()