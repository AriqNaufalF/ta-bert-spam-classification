import re, unicodedata, html
from unidecode import unidecode

# ---------------------------------------------------------------------------
# Tabel pemetaan: huruf kapital kecil (small capital / phonetic) → ASCII besar
# Sumber: Unicode Phonetic Extensions (U+1D00-U+1D7F), IPA Extensions (U+0250-U+02AF),
#         Latin Extended-B, Letterlike Symbols (U+A730-U+A731).
# ---------------------------------------------------------------------------
_SMALL_CAPS_TABLE = str.maketrans({
    # Phonetic Extensions (U+1D00-U+1D22)
    '\u1D00': 'A',  # ᴀ LATIN LETTER SMALL CAPITAL A
    '\u1D03': 'B',  # ᴃ LATIN LETTER SMALL CAPITAL BARRED B
    '\u1D04': 'C',  # ᴄ LATIN LETTER SMALL CAPITAL C
    '\u1D05': 'D',  # ᴅ LATIN LETTER SMALL CAPITAL D
    '\u1D06': 'D',  # ᴆ LATIN LETTER SMALL CAPITAL ETH
    '\u1D07': 'E',  # ᴇ LATIN LETTER SMALL CAPITAL E
    '\u1D0A': 'J',  # ᴊ LATIN LETTER SMALL CAPITAL J
    '\u1D0B': 'K',  # ᴋ LATIN LETTER SMALL CAPITAL K
    '\u1D0C': 'L',  # ᴌ LATIN LETTER SMALL CAPITAL L WITH STROKE
    '\u1D0D': 'M',  # ᴍ LATIN LETTER SMALL CAPITAL M
    '\u1D0E': 'N',  # ᴎ LATIN LETTER SMALL CAPITAL REVERSED N
    '\u1D0F': 'O',  # ᴏ LATIN LETTER SMALL CAPITAL O
    '\u1D18': 'P',  # ᴘ LATIN LETTER SMALL CAPITAL P
    '\u1D19': 'R',  # ᴙ LATIN LETTER SMALL CAPITAL REVERSED R
    '\u1D1A': 'R',  # ᴚ LATIN LETTER SMALL CAPITAL TURNED R
    '\u1D1B': 'T',  # ᴛ LATIN LETTER SMALL CAPITAL T
    '\u1D1C': 'U',  # ᴜ LATIN LETTER SMALL CAPITAL U
    '\u1D20': 'V',  # ᴠ LATIN LETTER SMALL CAPITAL V
    '\u1D21': 'W',  # ᴡ LATIN LETTER SMALL CAPITAL W
    '\u1D22': 'Z',  # ᴢ LATIN LETTER SMALL CAPITAL Z
    # IPA Extensions (U+0250-U+02AF)
    '\u0262': 'G',  # ɢ LATIN LETTER SMALL CAPITAL G
    '\u026A': 'I',  # ɪ LATIN LETTER SMALL CAPITAL I
    '\u0274': 'N',  # ɴ LATIN LETTER SMALL CAPITAL N
    '\u0280': 'R',  # ʀ LATIN LETTER SMALL CAPITAL R
    '\u028F': 'Y',  # ʏ LATIN LETTER SMALL CAPITAL Y
    '\u0299': 'B',  # ʙ LATIN LETTER SMALL CAPITAL B
    '\u029C': 'H',  # ʜ LATIN LETTER SMALL CAPITAL H
    '\u029F': 'L',  # ʟ LATIN LETTER SMALL CAPITAL L
    # Letterlike Symbols
    '\uA730': 'F',  # ꜰ LATIN LETTER SMALL CAPITAL F
    '\uA731': 'S',  # ꜱ LATIN LETTER SMALL CAPITAL S
})

# ---------------------------------------------------------------------------
# Tabel pemetaan: huruf dalam kotak/lingkaran negatif (Enclosed Alphanumeric)
# Blok U+1F170-U+1F18D: NEGATIVE SQUARED LATIN CAPITAL LETTER A-Z
# ---------------------------------------------------------------------------
_ENCLOSED_ALPHA_TABLE = str.maketrans({
    chr(0x1F170 + i): chr(ord('A') + i)
    for i in range(26)  # 🅰(A) sampai 🆉(Z)
})

def normalize_unicode_stylized(text: str) -> str:
    """
    Args:
        text: Teks yang akan dinormalisasi

    Returns:
        Teks yang sudah dinormalisasi ke ASCII standar
    """
    # Langkah 1: Pemetaan enclosed alphanumeric (sebelum NFKC agar tidak hilang)
    text = text.translate(_ENCLOSED_ALPHA_TABLE)

    # Langkah 2: NFKC — fullwidth, mathematical bold, superscript, dll.
    text = unicodedata.normalize("NFKC", text)

    # Langkah 3: Hapus combining diacritical marks (̽ ͟ ͓ dan sejenisnya)
    # NFD memecah karakter + diakritik menjadi terpisah, lalu buang kategori 'M'
    nfd = unicodedata.normalize("NFD", text)
    text = "".join(c for c in nfd if unicodedata.category(c)[0] != 'M')

    # Langkah 4: Pemetaan small capital / phonetic letters (ᴛ → T, ʙ → B, dll.)
    text = text.translate(_SMALL_CAPS_TABLE)

    return text

def transliterate_exotic_scripts(text: str) -> str:
    """
    Transliterasi karakter non-ASCII non-emoji yang tersisa ke ASCII.

    Menangani karakter eksotis seperti aksara Limbu, Javanese, Tai Viet,
    Canadian Syllabics yang digunakan spammer sebagai kamuflase.
    Contoh: '᥇ꪖꪻ᥅ꫀᔰᦔ' → 'batreqd' (approx)

    Menggunakan library `unidecode` sebagai fallback untuk karakter
    yang tidak tertangani oleh aturan-aturan sebelumnya.
    Emoji (unicode category So/Cs yang umum dipakai) dibiarkan apa adanya.

    Args:
        text: Teks yang masih mengandung karakter non-ASCII

    Returns:
        Teks dengan karakter eksotis yang sudah ditransliterasi
    """
    result = []
    for ch in text:
        if ord(ch) < 128:
            # ASCII murni — simpan apa adanya
            result.append(ch)
        elif unicodedata.category(ch) in ('So', 'Sm', 'Sk', 'Zs'):
            # Symbol / space variants — simpan (termasuk emoji non-letter)
            result.append(ch)
        elif unicodedata.category(ch).startswith('L') and ord(ch) > 127:
            # Huruf non-ASCII yang belum tertangani → transliterasi
            transliterated = unidecode(ch)
            result.append(transliterated if transliterated.strip() else ch)
        else:
            result.append(ch)
    return "".join(result)


_DIGIT_LEET_MAP: dict[str, str] = {
    '0': 'O',
    '1': 'I',
    '3': 'E',
    '4': 'A',
    '5': 'S',
    '6': 'G',
    '7': 'T',
    '8': 'B',
    '9': 'G',
}

_SYMBOL_LEET_TABLE = str.maketrans({
    '@': 'A',
    '$': 'S',
    '!': 'I',
    '+': 'T',
})


def normalize_leetspeak(text: str) -> str:
    """
    Normalisasi kata leetspeak (mengganti huruf standar dengan angka atau simbol)
    Strategi leet dibagi dua agar simbol trailing (BOTAK!?) tidak ikut diubah:
    1. DIGIT LEET — digit (0-9) yang menggantikan huruf di TENGAH kata. 
    Diterapkan hanya bila digit diapit huruf di kiri DAN kanan dalam token;

        - Contoh: T0GEL → TOGEL, TAKJUB4D → TAKJUBAD
        - Contoh tidak diubah: x1000, 37jt, 2024

    2. SIMBOL LEET — @, $, !, + yang menggantikan huruf di AWAL kata.
    
    Diterapkan hanya bila simbol berada tepat sebelum huruf di batas kata, sehingga tanda baca trailing tidak tersentuh.
    
        - Contoh: !KLAN → IKLAN, @NGGOTA → ANGGOTA
    Args:
        text: Teks yang mungkin mengandung leetspeak

    Returns:
        Teks dengan leetspeak yang sudah dinormalisasi
    """
    # --- Langkah 1: digit leet HANYA bila diapit huruf di kiri dan kanan ---
    def _replace_digit_leet(m: re.Match) -> str:
        token = m.group(0)
        chars = list(token)
        result = []
        for i, ch in enumerate(chars):
            if ch in _DIGIT_LEET_MAP:
                # Cek apakah ada huruf di sebelah kiri
                has_alpha_left  = any(c.isalpha() for c in chars[:i])
                # Cek apakah ada huruf di sebelah kanan
                has_alpha_right = any(c.isalpha() for c in chars[i + 1:])
                if has_alpha_left and has_alpha_right:
                    result.append(_DIGIT_LEET_MAP[ch])
                else:
                    result.append(ch)
            else:
                result.append(ch)
        return "".join(result)

    text = re.sub(r'\S+', _replace_digit_leet, text)

    # --- Langkah 2: simbol leet HANYA di awal kata (sebelum huruf) ---
    # (?<!\w) = tidak didahului karakter kata (word boundary kiri)
    # [@$!+]  = simbol leet
    # (?=[A-Za-z]) = diikuti langsung oleh huruf
    text = re.sub(
        r'(?<!\w)[@$!+](?=[A-Za-z])',
        lambda m: m.group(0).translate(_SYMBOL_LEET_TABLE),
        text,
    )

    return text


def collapse_dotted_letters(text: str) -> str:
    """
    Menggabungkan huruf/digit yang dipisahkan titik menjadi kata utuh

    Spammer kadang menulis 'B.B.C.A' untuk menghindari filter kata kunci. Pola yang ditangkap: 1 karakter huruf/digit, titik, berulang ≥ 3 kali.

    Contoh: 'B.B.C.A' → 'BBCA'
    Args:
        text: Teks yang mungkin mengandung huruf berpisah titik

    Returns:
        Teks dengan pola titik yang sudah digabung
    """
    pattern = re.compile(r'(?<!\w)([A-Za-z\d]{1,2})(?:\.[A-Za-z\d]{1,2}){2,}(?![\w.])')

    def _join_dots(m: re.Match) -> str:
        return m.group(0).replace('.', '')

    return pattern.sub(_join_dots, text)


def collapse_spaced_letters(text: str) -> str:
    """
    Menggabungkan chunk huruf/digit yang dipisahkan spasi tunggal menjadi kata utuh

    Spammer sering menulis kata dengan spasi antar huruf untuk menghindari filter kata kunci. Setelah normalisasi small-caps, huruf kecil palsu sudah dikonversi ke uppercase sehingga aman dideteksi.

    Contoh: 'W I N' → 'WIN'

    Strategi (token-scan):
    - Split teks menjadi token berdasarkan spasi.
    - Identifikasi run yang seluruh token-nya adalah 1-2 karakter huruf/digit UPPERCASE.
    - Gabungkan run ≥ 3 token; run pendek dibiarkan.
    Args:
        text: Teks yang mungkin mengandung huruf-huruf berpisah

    Returns:
        Teks dengan huruf-huruf berpisah yang sudah digabung
    """
    chunk_re = re.compile(r'^[A-Z\d]{1,2}$')
    tokens = text.split(" ")
    result_tokens: list[str] = []
    i = 0
    while i < len(tokens):
        run_start = i
        while i < len(tokens) and chunk_re.match(tokens[i]):
            i += 1
        run_end = i
        run_len = run_end - run_start
        if run_len >= 3:
            result_tokens.append("".join(tokens[run_start:run_end]))
        else:
            result_tokens.extend(tokens[run_start:run_end])
        if i == run_start:
            result_tokens.append(tokens[i])
            i += 1
    return " ".join(result_tokens)

def replace_emails(text: str) -> str:
    """
    Args:
        text: Teks yang mengandung alamat email

    Returns:
        Teks dengan alamat email yang sudah diganti [EMAIL]
    """
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return re.sub(email_pattern, '[EMAIL]', text)

def replace_timestamps(text: str) -> str:
    """
    Timestamp seperti '07:01' atau '07:01:01' sering muncul pada komentar YouTube
    yang merujuk ke titik waktu di video dan tidak mengandung informasi semantik
    untuk klasifikasi spam.

    Args:
        text: Teks yang mungkin mengandung timestamp

    Returns:
        Teks dengan timestamp yang sudah diganti [TIMESTAMP]
    """
    return re.sub(r'\b\d{1,2}:\d{2}(?::\d{2})?\b', "[TIMESTAMP]", text)

def replace_urls(text: str) -> str:
    """
    Args:
        text: Teks yang mengandung URL

    Returns:
        Teks dengan URL yang sudah diganti [URL]
    """
    return re.sub(r"http[s]?://\S+|www\.\S+", "[URL]", text)

def replace_mentions(text: str) -> str:
    """
    Args:
        text: Teks yang mengandung mention

    Returns:
        Teks dengan mention yang sudah diganti [USER]
    """
    return re.sub(r'@\w+', '[USER]', text)

def replace_hashtag(text) -> str:
    """
    Args:
        text: Teks yang akan dibersihkan

    Returns:
        Teks dengan hashtag yang sudah diganti [HASHTAG]
    """
    return re.sub(r'#\w+', '[HASHTAG]', text)


def remove_html_tags(text):
    """
    Args:
        text: Teks yang akan dibersihkan

    Returns:
        Teks yang sudah bersih dari tag HTML
    """
    text = html.unescape(text)
    pattern_a_tag = re.compile(r'<a\s+[^>]*href=[\"\']?[^\"\'>\s]+[\"\']?[^>]*>(.*?)</a>', re.IGNORECASE)
    text = pattern_a_tag.sub(r'[URL] \1', text)

    pattern_all_tags = re.compile(r'<[^>]+>')
    text = pattern_all_tags.sub(' ', text)

    return text


def clean_whitespace(text: str) -> str:
    """
    Args:
        text: Teks yang akan dibersihkan

    Returns:
        Teks yang sudah bersih dari whitespace berlebih
    """
    return re.sub(r"\s+", " ", text).strip()


# Fungsi pipeline preprocessing utama untuk membersihkan teks sebelum tokenisasi BERT

# Menangani berbagai teknik obfuskasi teks yang digunakan spammer:

# Urutan preprocessing:
# 1. remove_html_tags: Menghapus tag HTML (misal: `<br>`, `<a href=...>`)
# 2. normalize_unicode_stylized:
#     - Enclosed alphanumeric emoji (🅿→P, 🆄→U)
#     - NFKC: fullwidth (Ｐ→P), mathematical bold (𝐀→A)
#     - Combining diacritics strip (P͟U͟L͟→PUL)
#     - Small/phonetic caps (ᴛ→T, ɪ→I, ʙ→B)
# 3. transliterate_exotic_scripts: script eksotis (᥇ꪖꪻ → approx ASCII)
# 4. collapse_dotted_letters: titik-pisah (B.B.C.A.4D → BBCA4D)
# 5. normalize_leetspeak: digit leet di TENGAH kata (T0GEL→TOGEL, TAKJUB4D→TAKJUBAD;
# digit di tepi dibiarkan: x1000→x1000, 37jt→37jt, TOGEL46→TOGEL46)
# 6. collapse_spaced_letters: spasi-pisah (W I N → WIN)
# 7. replace_urls: URL → [URL]
# 8. replace_emails: email → [EMAIL]
# 9. replace_timestamps: 07:01 → "[TIMESTAMP]"
# 10. replace_mentions: @user → [USER]
# 11. replace_hashtag: #hashtag → [HASHTAG]
# 12. clean_whitespace: spasi berlebih

def preprocess_text(text: str) -> str:
    """
    Args:
        text: Teks mentah yang akan diproses

    Returns:
        Teks yang sudah bersih dan siap untuk tokenisasi
    """
    if not isinstance(text, str):
        return ""

    text = remove_html_tags(text)
    text = normalize_unicode_stylized(text)
    text = transliterate_exotic_scripts(text)
    text = collapse_dotted_letters(text)
    text = normalize_leetspeak(text)
    text = collapse_spaced_letters(text)
    text = replace_urls(text)
    text = replace_emails(text)
    text = replace_timestamps(text)
    text = replace_mentions(text)
    text = replace_hashtag(text)
    text = clean_whitespace(text)

    return text