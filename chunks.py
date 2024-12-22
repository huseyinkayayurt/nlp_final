from typing import List, Dict

def collect_chunks_with_indices(data: List[Dict]) -> (List[str], Dict[int, int]):
    """
    Tüm chunk'ları tek bir listeye toplar ve doğru chunk indekslerini kaydeder.

    Args:
        data (List[Dict]): 1000 elemanlı veri kümesi.

    Returns:
        List[str]: Tüm chunk'ların yer aldığı 5000 elemanlı liste.
        Dict[int, int]: Her soru için doğru chunk'ın yeni indeksini gösteren sözlük.
    """
    all_chunks = []
    correct_indices = {}
    current_index = 0

    for idx, item in enumerate(data):
        chunks = item["chunks"]
        correct_intro_idx = int(item["correct_intro_idx"])  # Doğru chunk indexi

        # Chunk'ları tüm chunk listesine ekle
        all_chunks.extend(chunks)

        # Doğru chunk indexini güncel indeksle eşleştir
        correct_indices[idx] = current_index + correct_intro_idx

        # Güncel indeksi artır
        current_index += len(chunks)

    return all_chunks, correct_indices
