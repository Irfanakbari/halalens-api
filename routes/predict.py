import time
import re
import os
from flask import Blueprint, request, jsonify
from services.cloud_storage import upload_to_storage
from services.cloud_vision import detect_text_uri
from thefuzz import process, fuzz
import google.generativeai as genai

# Konfigurasi Gemini API
genai.configure(api_key="AIzaSyDfdp9pWCEDT_4TkxmZWJ4iA0LYj43RAa0")

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
)

predict_bp: Blueprint = Blueprint("predict", __name__)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Daftar bahan makanan non-halal
non_halal_ingredients = [
    # Daging babi dan turunannya
    "babi", "pig", "pork", "ham", "bacon", "lard", "sow", "swine",
    "hog", "boar", "suckling pig", "pork chop", "pork loin",

    # Sosis dan olahan babi
    "chorizo", "salami", "prosciutto", "mortadella",
    "capicola", "pancetta", "guanciale",

    # Lemak & minyak babi
    "lard", "schmaltz", "shortening", "pork fat",

    # Gelatin & zat aditif dari hewan non-halal
    "gelatin", "gelatine", "rennet", "carmine", "E441", "E120",
    "enzymes", "lipase", "trypsin", "rennin", "pepsin",

    # Alkohol dan minuman haram
    "alkohol", "alcohol", "beer", "wine", "whiskey", "vodka", "gin",
    "rum", "brandy", "tequila", "cider", "sake", "mirin",

    # Darah dan bagian hewan yang dilarang
    "darah", "blood", "black pudding", "blood sausage",

    # Hewan non-halal lainnya
    "dog", "cat", "horse meat", "donkey meat", "frog", "turtle",
    "snail", "lizard", "alligator", "crocodile", "kangaroo",

    # Makanan laut yang tidak diperbolehkan menurut sebagian pendapat
    "shark", "eel", "stingray", "shellfish", "lobster", "crab",
    "clams", "mussels", "oysters", "scallops",

    # Cuka yang difermentasi dengan alkohol
    "wine vinegar", "malt vinegar",

    # Bahan kimia yang berasal dari hewan
    "shellac", "E904", "castoreum", "lanolin", "E913",
    "keratin", "L-cysteine", "E920", "stearic acid", "E570",
    "glycerol", "E422", "monoglycerides", "diglycerides", "E471",
    "lecithin", "E322", "animal rennet"
]


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def clean_text(text):
    """Menghapus karakter aneh dan mengonversi teks ke huruf kecil."""
    return re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()


def detect_non_halal(text, threshold=85):
    detected = []
    words = clean_text(text).split()

    for word in words:
        if len(word) < 3:  # Abaikan kata yang terlalu pendek
            continue

        # Cek hanya jika panjang kata mirip
        best_match, score = process.extractOne(
            word, non_halal_ingredients, scorer=fuzz.ratio
        )

        # Tambahkan aturan bahwa panjang kata tidak boleh berbeda lebih dari 2 karakter
        if score >= threshold and abs(len(word) - len(best_match)) <= 2:
            detected.append({"word": word, "match": best_match, "score": score})

    return detected


@predict_bp.route('/predict', methods=['POST'])
def predict():
    try:
        start_time = time.time()

        if 'file' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No Selected File"}), 400

        if file and allowed_file(file.filename):
            upload_start_time = time.time()
            uri = upload_to_storage(file)
            upload_time = time.time() - upload_start_time

            ocr_start_time = time.time()
            ocr_text = detect_text_uri(uri["uri"])
            ocr_time = time.time() - ocr_start_time

            # Bersihkan teks sebelum AI processing
            cleaned_text = clean_text(ocr_text)

            # Gunakan AI hanya jika teks OCR terlalu berantakan
            if len(cleaned_text.split()) > 20:
                response = model.generate_content([
                    "Extract only food ingredients from the following text:",
                    "Please remove unnecessary word in food commposition, and extract only food composition.",
                    "input: Food Type Miso contents G raw material name and content purified water, soybean, "
                    "sun salt, wheat, alcohol, alcohol, long -sized soybean, wheat containing expiration date, mold, "
                    "and day of packaging, leaded polyethylene inner, container polypropylene, lactation 제조원 및 판매원 "
                    "샘표식품 주식회사 본사 서울시 중구 충무로 공장 충청북도 영동군 용산면 용심로 반품 및 교환처 본사 및 구입처 본 제품은 소비자 기본법의 일반적 소비자 분쟁해결 기준에"
                    "의거, 교환 등 보상을 받을 수 있습니다 사용 및 보관시 주의사항 Please keep the direct sunlight at room temperature and "
                    "keep it refrigerated after opening. Poe is burdened SGS by Supplier Plastic PP container, lid, "
                    "label PP lead paper",
                    "output: purified water, soybean, sun salt, wheat, alcohol, alcohol, long -sized soybean, wheat",
                    "input: Contents G CJ Premial Snow White Ribs Seasoned Ribs Seasoned Ribs Seasoned Yeonmu -eup"
                    "Jukbon Gil Foods Report Non -Eup Status Report Number Consumer Number Consumer Low room "
                    "temperature storage, and after opening, refrigerated in the refrigerator, container glass Amino "
                    "acid surface -to -soybeans, brewing, brewing, brewed, brewed, brewed, soybeans, "
                    "refined stations, sugar, other fructose, bafure pear domestic, vitamin C, purified water, "
                    "onion pure Chinese, minced garlic apple pure apple Paemit, Caramel Soldier III, Pepper Pepper, "
                    "Pepper Primulates, Mountain Controls, Zantan Sword, Citrus Extract",
                    "output: glass Amino acid surface -to -soybeans, brewing, brewed, soybeans, refined stations, "
                    "sugar, other fructose, bafure pear domestic, vitamin C, purified water, onion pure Chinese, "
                    "minced garlic apple pure apple Paemit, Caramel Soldier III, Pepper, Pepper Primulates, "
                    "Mountain Controls, Zantan Sword, Citrus Extract",
                    "input: PEACH PEACH BON MAT HAICHED with Peach Sacs Juice RP Peach Each week, Fi, Chungnam -gu,"
                    "Cheonan -si, Chungnam -gu, Cheon -gu, Gyeongbuk COE Consumption Economic Dogs Dogs Economic "
                    "Water, Dangrup, Grape Granal Chinese, Grape Located Italy Asan, Mixed Contents Grape Grape Juice "
                    "Restorous Standards, Citric acid, Synthetic Fragrance Vine If the variable Pyeongchang is "
                    "damaged or the contents are altered, do not drink it. Please do not drink it after opening. "
                    "Senior Products Exchange Customer Counseling Office and Each Purchasing Products can be "
                    "exchanged or compensated by the FTC notice. It is manufactured in the contents that natural and "
                    "the environment is beautifully charged nitrogen cleansing www tbcokr As it is KCAL standard, "
                    "it may vary depending on the amount of calories required rawberry onbon This is an extract of "
                    "Haital Crushed Strawberry Ju ML Spessifikasi M Kode Barang rp dlatte",
                    "output: Water, Dangrup, Grape Granal Chinese, Grape Located Italy Asan, Mixed Contents Grape "
                    "Grape Juice Restorous Standards, Citric acid, Synthetic Fragrance Vine",
                    f"input: {ocr_text}",
                    "output: "
                ])
                processed_text = response.text
            else:
                processed_text = cleaned_text  # Jika sudah bersih, pakai langsung

            detect_start_time = time.time()
            detected_non_halal = detect_non_halal(processed_text)
            detect_time = time.time() - detect_start_time

            return jsonify({
                "ocr_text": ocr_text,
                "cleaned_text": processed_text,
                "detected_non_halal": detected_non_halal,
                "image": uri["link"],
                "timings": {
                    "upload_time": upload_time,
                    "ocr_time": ocr_time,
                    "detection_time": detect_time
                }
            }), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

    return jsonify({"error": "Invalid File Type"}), 400
