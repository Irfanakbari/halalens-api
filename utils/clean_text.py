def clean_text(text):
    # Ganti karakter yang tidak diinginkan dengan spasi
    cleaned_text = ''.join(char if char.isalpha() or char.isspace() or char == ',' else ' ' for char in text)

    # Hapus spasi berlebihan dan ubah menjadi lowercase
    cleaned_text = ' '.join(cleaned_text.split()).lower()

    return cleaned_text
