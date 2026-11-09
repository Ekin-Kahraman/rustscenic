"""Keep the separately packaged description usable outside GitHub."""
from html.parser import HTMLParser
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]


def test_package_readme_is_current_and_has_portable_links():
    subprocess.run([sys.executable, str(ROOT / "scripts/render_pypi_readme.py"), "--check"], check=True)
    description = (ROOT / "README.pypi.md").read_text()
    urls = re.findall(r'\]\(([^\s)]+)\)', description)

    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            urls.extend(value for key, value in attrs if key in {"src", "href", "srcset"})

    Links().feed(description)
    assert len(urls) > 50
    assert all(urlsplit(url).scheme in {"https", "http", "mailto"} for url in urls)
    images = re.findall(r'!\[[^\[\]\n]*\]\(([^\s)]+)\)', description)
    assert images
    assert all(not url.startswith("https://github.com/Ekin-Kahraman/rustscenic/blob/") for url in images)
    assert "https://colab.research.google.com/github/Ekin-Kahraman/rustscenic/blob/main/" not in description
    assert 'readme = "README.pypi.md"' in (ROOT / "pyproject.toml").read_text()
