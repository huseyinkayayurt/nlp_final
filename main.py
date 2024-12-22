from data_loader import load_data

def main():
    """
    Ana fonksiyon, veri setini yükler ve bir örneği konsola yazdırır.
    """
    file_path = "data_set/train.csv"
    data = load_data(file_path, size=1000)

    if data:
        print(f"İşlenmiş veri kümesinden ilk eleman:\n{data[0]}")
        print(f"Chunks: {data[0]['chunks']}")
        print(f"Toplam eleman sayısı: {len(data)}")
    else:
        print("Veri kümesi yüklenemedi.")

if __name__ == "__main__":
    main()
