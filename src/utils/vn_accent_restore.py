import torch
from transformers import AutoModelForTokenClassification
from enum import Enum

# Cấu hình
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "saeliddp/distilbert-viet-diacritic-restoration"
# MODEL_NAME = "/content/drive/MyDrive/VN-ACCENT-RESTORE/models/finetuned-chunk-001"

# Model sẽ được load lazy khi cần
model = None

# id cho các ký tự nguồn (source chars) – phải trùng với model đã train
SRC_CHARS = "abcdeghiklmnopqrstuvxy"

SRC_CHAR_TO_ID = {c: i + 4 for i, c in enumerate(SRC_CHARS)}
SRC_CHAR_TO_ID["^"] = 0   # WORD_START
SRC_CHAR_TO_ID["#"] = 1   # NUMERIC
SRC_CHAR_TO_ID["$"] = 2   # UNKNOWN
SRC_CHAR_TO_ID[","] = 3   # PAD

ID_TO_SRC_CHAR = {v: k for k, v in SRC_CHAR_TO_ID.items()}


class Mark(Enum):
    # không dấu phụ (nguyên âm thường)
    NONE_NONE  = 0
    NONE_GRAVE = 1   # huyền
    NONE_ACUTE = 2   # sắc
    NONE_HOOK  = 3   # hỏi
    NONE_TILDE = 4   # ngã
    NONE_DOT   = 5   # nặng

    # chữ d có gạch ngang
    CROSS_NONE = 6

    # mũ: â, ê, ô + các thanh
    HAT_NONE   = 7
    HAT_GRAVE  = 8
    HAT_ACUTE  = 9
    HAT_HOOK   = 10
    HAT_TILDE  = 11
    HAT_DOT    = 12

    # nguyên âm có “râu”: ơ, ư + các thanh
    TAIL_NONE  = 13
    TAIL_GRAVE = 14
    TAIL_ACUTE = 15
    TAIL_HOOK  = 16
    TAIL_TILDE = 17
    TAIL_DOT   = 18

    # ă + các thanh
    SWOOP_NONE  = 19
    SWOOP_GRAVE = 20
    SWOOP_ACUTE = 21
    SWOOP_HOOK  = 22
    SWOOP_TILDE = 23
    SWOOP_DOT   = 24

ID_TO_MARK = {m.value: m for m in Mark}

CD_TO_CHAR = {}

# a/e/i/o/u + 5 thanh
for base, graves, acutes, hooks, tildes, dots in [
    ("a", "à", "á", "ả", "ã", "ạ"),
    ("e", "è", "é", "ẻ", "ẽ", "ẹ"),
    ("i", "ì", "í", "ỉ", "ĩ", "ị"),
    ("o", "ò", "ó", "ỏ", "õ", "ọ"),
    ("u", "ù", "ú", "ủ", "ũ", "ụ"),
]:
    CD_TO_CHAR[(base, Mark.NONE_GRAVE)] = graves
    CD_TO_CHAR[(base, Mark.NONE_ACUTE)] = acutes
    CD_TO_CHAR[(base, Mark.NONE_HOOK)]  = hooks
    CD_TO_CHAR[(base, Mark.NONE_TILDE)] = tildes
    CD_TO_CHAR[(base, Mark.NONE_DOT)]   = dots

# đ
CD_TO_CHAR[("d", Mark.CROSS_NONE)] = "đ"

# â, ê, ô + thanh
for base, plain, graves, acutes, hooks, tildes, dots in [
    ("a", "â", "ầ", "ấ", "ẩ", "ẫ", "ậ"),
    ("e", "ê", "ề", "ế", "ể", "ễ", "ệ"),
    ("o", "ô", "ồ", "ố", "ổ", "ỗ", "ộ"),
]:
    CD_TO_CHAR[(base, Mark.HAT_NONE)]   = plain
    CD_TO_CHAR[(base, Mark.HAT_GRAVE)]  = graves
    CD_TO_CHAR[(base, Mark.HAT_ACUTE)]  = acutes
    CD_TO_CHAR[(base, Mark.HAT_HOOK)]   = hooks
    CD_TO_CHAR[(base, Mark.HAT_TILDE)]  = tildes
    CD_TO_CHAR[(base, Mark.HAT_DOT)]    = dots

# ơ, ư + thanh
for base, plain, graves, acutes, hooks, tildes, dots in [
    ("o", "ơ", "ờ", "ớ", "ở", "ỡ", "ợ"),
    ("u", "ư", "ừ", "ứ", "ử", "ữ", "ự"),
]:
    CD_TO_CHAR[(base, Mark.TAIL_NONE)]   = plain
    CD_TO_CHAR[(base, Mark.TAIL_GRAVE)]  = graves
    CD_TO_CHAR[(base, Mark.TAIL_ACUTE)]  = acutes
    CD_TO_CHAR[(base, Mark.TAIL_HOOK)]   = hooks
    CD_TO_CHAR[(base, Mark.TAIL_TILDE)]  = tildes
    CD_TO_CHAR[(base, Mark.TAIL_DOT)]    = dots

# ă + thanh
for base, plain, graves, acutes, hooks, tildes, dots in [
    ("a", "ă", "ằ", "ắ", "ẳ", "ẵ", "ặ"),
]:
    CD_TO_CHAR[(base, Mark.SWOOP_NONE)]   = plain
    CD_TO_CHAR[(base, Mark.SWOOP_GRAVE)]  = graves
    CD_TO_CHAR[(base, Mark.SWOOP_ACUTE)]  = acutes
    CD_TO_CHAR[(base, Mark.SWOOP_HOOK)]   = hooks
    CD_TO_CHAR[(base, Mark.SWOOP_TILDE)]  = tildes
    CD_TO_CHAR[(base, Mark.SWOOP_DOT)]    = dots


def rediacritize_char(base_char: str, mark_id: int) -> str:
    """
    base_char: chữ thường không dấu (a..z)
    mark_id: số 0..24 từ model
    """
    mark = ID_TO_MARK.get(int(mark_id), Mark.NONE_NONE)
    key = (base_char.lower(), mark)
    if key in CD_TO_CHAR:
        return CD_TO_CHAR[key]
    # không có mapping đặc biệt -> trả lại chữ gốc
    return base_char

def encode_sentence_ascii(sent: str):
    """
    Trả về:
      ids: list[int]  - sequence đưa vào model
      base_chars: list[str] - cùng độ dài, là ký tự base tương ứng (hoặc None cho '^')
    """
    ids = []
    base_chars = []

    for word in sent.split():
        # token WORD_START
        ids.append(SRC_CHAR_TO_ID["^"])
        base_chars.append(None)  # không phải ký tự thật

        for ch in word:
            ch_low = ch.lower()
            if ch_low in SRC_CHAR_TO_ID:
                ids.append(SRC_CHAR_TO_ID[ch_low])
                base_chars.append(ch)
            elif ch_low.isdigit():
                ids.append(SRC_CHAR_TO_ID["#"])
                base_chars.append(ch)
            else:
                ids.append(SRC_CHAR_TO_ID["$"])
                base_chars.append(ch)

    return ids, base_chars


def restore_diacritics_ascii(sentence: str) -> str:
    # B1: mã hoá
    input_ids, base_chars = encode_sentence_ascii(sentence)

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
    attn_mask = torch.ones_like(input_tensor)

    with torch.no_grad():
        logits = model(input_ids=input_tensor, attention_mask=attn_mask).logits
        pred_ids = logits.argmax(-1)[0].cpu().tolist()  # (seq_len,)

    # B2: ghép lại theo base_chars
    out_chars = []
    for ch, mark_id in zip(base_chars, pred_ids):
        if ch is None:
            # đây là token '^' -> chuyển thành khoảng trắng giữa các từ
            if out_chars and out_chars[-1] != " ":
                out_chars.append(" ")
            continue
        # ch là ký tự gốc trong câu (không dấu)
        if ch.lower() in "adeiouy":
            out_chars.append(rediacritize_char(ch, mark_id))
        else:
            # phụ âm, số, ký tự đặc biệt – giữ nguyên
            out_chars.append(ch)

    return "".join(out_chars).strip()


def load_model():
    """Load model một lần và cache lại"""
    global model
    if model is None:
        print("Đang tải model...")
        try:
            # Thử load với safetensors trước (an toàn hơn và không cần torch >= 2.6)
            model = AutoModelForTokenClassification.from_pretrained(
                MODEL_NAME,
                use_safetensors=True
            )
        except Exception as e:
            # Nếu không có safetensors, thử load bình thường
            print(f"Không thể load với safetensors: {e}")
            print("Thử load với phương thức khác...")
            model = AutoModelForTokenClassification.from_pretrained(
                MODEL_NAME,
                use_safetensors=False
            )
        model.to(DEVICE)
        model.eval()
        print(f"Model đã được tải. Device: {DEVICE}")
        print(f"num_labels = {model.config.num_labels}")  # nên là 25 hoặc 26
        print(f"vocab_size = {model.config.vocab_size}")  # 26
    return model


def restore_diacritics(sentence: str) -> str:
    """
    Hàm chính để phục hồi dấu cho câu tiếng Việt không dấu.
    
    Args:
        sentence: Câu tiếng Việt không dấu (ví dụ: "xin chao cac ban")
    
    Returns:
        Câu đã được phục hồi dấu (ví dụ: "xin chào các bạn")
    """
    load_model()  # Đảm bảo model đã được load
    return restore_diacritics_ascii(sentence)


if __name__ == "__main__":
    # Demo
    test_sentences = [
        "xin chao cac ban",
        "toi ten la Nguyen Van A",
        "hom nay troi rat dep",
        "ban co khoe khong"
    ]
    
    print("=" * 50)
    print("DEMO: Phục hồi dấu tiếng Việt")
    print("=" * 50)
    print()
    
    for s in test_sentences:
        result = restore_diacritics(s)
        print(f"Input : {s}")
        print(f"Output: {result}")
        print()
