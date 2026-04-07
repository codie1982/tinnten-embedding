"""
HTML -> Markdown Clean donusumu.
tinnten-fetcher/services/content_metadata.py ile ayni mantik.
URL import akisinda ham HTML yerine temiz markdown chunk'lanir.
"""

import re
from typing import Optional

from bs4 import BeautifulSoup
from markdownify import markdownify as md


def html_to_clean_markdown(html: str, url: Optional[str] = None) -> str:
    """Ham HTML'i temiz markdown'a donusturur."""
    if not html or not isinstance(html, str):
        return ""

    # BeautifulSoup ile parse et, script/style/nav temizle
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["script", "style", "noscript", "nav", "footer", "header", "iframe"]):
        tag.decompose()

    # Body varsa sadece body icerigini al
    body = soup.find("body")
    clean_html = str(body) if body else str(soup)

    # markdownify ile HTML -> Markdown donusumu
    raw_markdown = md(clean_html, heading_style="ATX", strip=["img"])

    return clean_markdown(raw_markdown)


def clean_markdown(md_text: str) -> str:
    """
    Markdown metnini temizler: gereksiz linkler, footer, tekrar eden satirlar kaldirilir.
    tinnten-fetcher/services/content_metadata.py/clean_markdown() ile uyumlu.
    """
    if not isinstance(md_text, str):
        return ""

    out = md_text.replace("\r\n", "\n").replace("\r", "\n").strip()

    # Regex temizlikleri
    out = re.sub(r"\(<\s*([^>]+)\s*>\)", r"(\1)", out, flags=re.I)
    out = re.sub(r"\b(https?):\/(?!\/)", r"\1://", out, flags=re.I)
    out = re.sub(r"^\s*\[\d+\]\s*", "", out, flags=re.M)
    out = re.sub(
        r"\[([^\]]*)\]\(\s*javascript:void\(0\)\s*\)",
        lambda m: (m.group(1) or "").strip(),
        out,
        flags=re.I,
    )
    out = re.sub(
        r"\[([^\]]+)\]\(\s*(#|\s*)\)",
        lambda m: (m.group(1) or "").strip(),
        out,
    )
    out = re.sub(r"\[\s*\]\(\s*[^)]+\s*\)", "", out)
    out = re.sub(
        r"\[([^\]]+)\]\(\s*[^)]+\s*\)",
        lambda m: "\n" + m.group(1) + "\n",
        out,
    )
    out = re.sub(r"!\[[^\]]*\]\(\s*[^)]+\s*\)", "", out)  # Gorselleri kaldir
    out = re.sub(r"\bhttps?://\S+", "", out, flags=re.I)  # Tek basina URL'leri kaldir

    # Bosluk normalizasyonu
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n[ \t]+", "\n", out)

    lines = [line.strip() for line in out.split("\n")]

    # Footer / noise kaliplari
    footer_or_noise = re.compile(
        "|".join(
            [
                r"^gizlilik politikasi$",
                r"^kvkk$",
                r"^kullanim kosullari$",
                r"^cerez politikasi$",
                r"^privacy policy$",
                r"^terms(?: and conditions)?$",
                r"^cookies?$",
                r"^contact$",
                r"(mah\.|mahalle|sokak|cad\.|no:|pk:|kat\b|daire\b|istanbul|ankara|izmir)",
                r"^copyright|^(c)",
            ]
        ),
        flags=re.I,
    )

    lines = [l for l in lines if l and not footer_or_noise.search(l) and len(l) > 2]

    # Tekrar eden satirlari kaldir
    seen, deduped = set(), []
    for l in lines:
        key = l.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(l)

    result = "\n".join(deduped)
    result = re.sub(r"\n{3,}", "\n\n", result).strip()
    return result
