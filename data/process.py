import csv


def process_dataset(file_path, tokenizer, max_length=512):
    processed_data = []

    with open(file_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Context'i chunk'lara ayır
            context = row["context"]
            split_points = eval(row["ctx_split_points"])
            chunks = [
                context[start:end].strip()
                for start, end in zip([0] + split_points, split_points + [len(context)])
            ]

            # Question ve chunk'ları birleştir
            inputs = tokenizer(
                row["question"],
                " ".join(chunks),
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )

            processed_data.append({
                "input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0),
                "labels": int(row["correct_intro_idx"]),
            })

    return processed_data
