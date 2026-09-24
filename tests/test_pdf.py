import pytest
from pypdf import PdfReader, PdfWriter

import pdf_reader
from pdf_reader import PDFError
from pdf_helper import make_text_pdf


def test_extract_text_from_generated_pdf(tmp_path):
    path = pdf_reader.save_upload("paper.pdf", make_text_pdf(["Solar irradiance forecasting with physics constraints.",
                                                               "Second page about LSTM baselines."]), tmp_path)
    info = pdf_reader.load_pdf(path)
    assert info["pages"] == 2
    assert "irradiance" in info["chunks"][0] and "LSTM" in info["chunks"][0]


def test_chunking_with_overlap():
    words = [f"w{i}" for i in range(4500)]
    chunks = pdf_reader.chunk_words(" ".join(words), size=2000, overlap=200)
    assert len(chunks) == 3
    assert chunks[1].split()[0] == "w1800"          # second chunk starts 200 words before the first ended
    assert chunks[-1].split()[-1] == "w4499"


def test_top_chunks_picks_relevant_ones_in_order():
    chunks = ["intro about cats", "solar forecasting physics model", "dogs and more dogs", "physics solar metrics"]
    picked = pdf_reader.top_chunks(chunks, "What solar physics model is used?", k=2)
    assert [i for i, _ in picked] == [1, 3]


@pytest.mark.parametrize("name, expected", [
    ("../../etc/passwd.pdf", "passwd.pdf"),
    ("my paper (final).PDF", "my_paper_final.pdf"),
    ("..\\..\\evil.pdf", "evil.pdf"),
    ("", "upload.pdf"),
])
def test_sanitize_filename(name, expected):
    assert pdf_reader.sanitize_filename(name) == expected


def test_upload_is_saved_only_inside_uploads(tmp_path):
    path = pdf_reader.save_upload("../../escape.pdf", make_text_pdf(["hello world text here ok"]), tmp_path / "uploads")
    assert path.parent == (tmp_path / "uploads").resolve()


def test_rejects_non_pdf(tmp_path):
    with pytest.raises(PDFError, match="Only .pdf"):
        pdf_reader.save_upload("notes.txt", b"hello", tmp_path)
    with pytest.raises(PDFError, match="not a valid PDF"):
        pdf_reader.save_upload("fake.pdf", b"MZ\x90 not a pdf", tmp_path)


def test_rejects_large_file(tmp_path):
    with pytest.raises(PDFError, match="20 MB"):
        pdf_reader.save_upload("big.pdf", b"%PDF" + b"0" * pdf_reader.MAX_BYTES, tmp_path)


def test_rejects_encrypted_pdf(tmp_path):
    plain = tmp_path / "plain.pdf"
    plain.write_bytes(make_text_pdf(["secret research text goes here"]))
    writer = PdfWriter(clone_from=PdfReader(plain))
    writer.encrypt(user_password="pw", algorithm="RC4-128")
    locked = tmp_path / "locked.pdf"
    with open(locked, "wb") as f:
        writer.write(f)
    with pytest.raises(PDFError, match="encrypted"):
        pdf_reader.load_pdf(locked)


def test_rejects_image_only_pdf(tmp_path):
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    blank = tmp_path / "scan.pdf"
    with open(blank, "wb") as f:
        writer.write(f)
    with pytest.raises(PDFError, match="No text found"):
        pdf_reader.load_pdf(blank)
