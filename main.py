from chunks import collect_chunks_with_indices
from data_loader import load_data

def main():
    """
    Ana fonksiyon, veri setini yükler ve bir örneği konsola yazdırır.
    """
    file_path = "data_set/train.csv"
    data = load_data(file_path, size=1000)

    if data:
        soru_id = 10
        print(f"soru {soru_id}: {data[soru_id]}")
        all_chunks, correct_indices = collect_chunks_with_indices(data)

        print(f"Toplam chunk sayısı: {len(all_chunks)}")  # 5000 bekleniyor
        print(f"İlk birkaç chunk: {all_chunks[:5]}")
        print(f"Doğru indeksler: {list(correct_indices.items())[:5]}")


        dogru_chunk_index = correct_indices[soru_id]
        print(f"Soru {soru_id} için doğru chunk: {all_chunks[dogru_chunk_index]}")
    else:
        print("Veri kümesi yüklenemedi.")

if __name__ == "__main__":
    main()
