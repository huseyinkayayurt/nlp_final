import csv
from typing import List, Dict


def load_data(file_path: str, size: int = 1000) -> List[Dict[str, str]]:
    """
    CSV dosyasını okuyarak, context eleman sayısı 5 olan ve belirtilen boyutta veri döner.

    Args:
        file_path (str): CSV dosyasının dosya yolu.
        size (int): İstenen veri boyutu (default: 1000).

    Returns:
        List[Dict[str, str]]: İşlenmiş veri kümesi.
    """
    dataset = []
    try:
        with open(file_path, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Context chunk'larını ayrıştır
                context = row["context"]
                split_points = eval(row["ctx_split_points"])  # String'i listeye dönüştür
                chunks = extract_chunks(context, split_points)

                # Eğer chunk sayısı 5 ise ekle
                if len(chunks) == 5:
                    row["chunks"] = chunks
                    dataset.append(row)

                # İstenen boyuta ulaşıldığında döngüden çık
                if len(dataset) >= size:
                    break
    except FileNotFoundError:
        print(f"Hata: Dosya bulunamadı - {file_path}")
    except Exception as e:
        print(f"Beklenmeyen bir hata oluştu: {e}")
    return dataset


def extract_chunks(context: str, split_points: List[int]) -> List[str]:
    """
    Context metnini verilen ayrılma noktalarına göre chunk'lara böler.

    Args:
        context (str): Metin.
        split_points (List[int]): Ayrılma noktalarının indeksleri.

    Returns:
        List[str]: Chunk'lar.
    """
    chunks = []
    start = 0
    for end in split_points:
        chunks.append(context[start:end].strip())  # Ayrılan metni ekle
        start = end
    # Son split_point'ten metnin sonuna kadar olan kısmı ekle
    if start < len(context):
        chunks.append(context[start:].strip())
    return chunks
