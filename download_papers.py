"""
Step 1: Download 50 papers from OpenReview and extract text.
Papers are from the 2025 NeurIPS AI4Mat track.
"""

import os
import json
import time
import requests
import fitz  # PyMuPDF

PAPER_URLS = [
    "https://openreview.net/pdf?id=0SPoKR8Xrk",
    "https://openreview.net/pdf?id=0uWNuJ1xtz",
    "https://openreview.net/pdf?id=12ZCZVKm7r",
    "https://openreview.net/pdf?id=24lzMGlvnq",
    "https://openreview.net/pdf?id=35aDuh7ndX",
    "https://openreview.net/pdf?id=3WZkuWlzmN",
    "https://openreview.net/pdf?id=3pAVbjWMXW",
    "https://openreview.net/pdf?id=4U2k4uw43B",
    "https://openreview.net/pdf?id=4Xh9oL5rH0",
    "https://openreview.net/pdf?id=57YLCp7n2V",
    "https://openreview.net/pdf?id=5OsnDm1CdX",
    "https://openreview.net/pdf?id=6pjxodugzO",
    "https://openreview.net/pdf?id=7brF4sMQq3",
    "https://openreview.net/pdf?id=7cbwuA5k0T",
    "https://openreview.net/pdf?id=7l75CbxtmC",
    "https://openreview.net/pdf?id=8JFITrNy3K",
    "https://openreview.net/pdf?id=9JSO4qf1RQ",
    "https://openreview.net/pdf?id=A21WF9M1Um",
    "https://openreview.net/pdf?id=AQkGpEMGWA",
    "https://openreview.net/pdf?id=Bg4Hn9Qq3w",
    "https://openreview.net/pdf?id=Cfj7uBu5dy",
    "https://openreview.net/pdf?id=Ciw6DbDa4U",
    "https://openreview.net/pdf?id=Ei3eF8B8XH",
    "https://openreview.net/pdf?id=EuACaJblk4",
    "https://openreview.net/pdf?id=Gzf8k2wPdF",
    "https://openreview.net/pdf?id=InZczCC8X1",
    "https://openreview.net/pdf?id=YKxwBMK8Nl",
    "https://openreview.net/pdf?id=a3LKICpDO2",
    "https://openreview.net/pdf?id=aECXy5Jgm4",
    "https://openreview.net/pdf?id=acfR6umMJt",
    "https://openreview.net/pdf?id=amn6lBDjXm",
    "https://openreview.net/pdf?id=auRe7zr32I",
    "https://openreview.net/pdf?id=bmgU7yWBeC",
    "https://openreview.net/pdf?id=cEgjPFdLvl",
    "https://openreview.net/pdf?id=cFTvHHXvt6",
    "https://openreview.net/pdf?id=ctyy8EJYQj",
    "https://openreview.net/pdf?id=dEtRvi7G5i",
    "https://openreview.net/pdf?id=dmeAH1hVR8",
    "https://openreview.net/pdf?id=e8bcQehZ15",
    "https://openreview.net/pdf?id=eUiZg9uUt4",
    "https://openreview.net/pdf?id=egi8g2U0ZX",
    "https://openreview.net/pdf?id=enQdbinvNd",
    "https://openreview.net/pdf?id=farKrjdsIH",
    "https://openreview.net/pdf?id=g6Sj1OFjAu",
    "https://openreview.net/pdf?id=gifMFKvAl5",
    "https://openreview.net/pdf?id=hFzjgQzoVU",
    "https://openreview.net/pdf?id=hQCdhenqre",
    "https://openreview.net/pdf?id=hk6iX4mg3B",
    "https://openreview.net/pdf?id=iFHaZzs6Kz",
    "https://openreview.net/pdf?id=j3aOU8Ahue",
]

def extract_paper_id(url):
    """Extract paper ID from OpenReview URL."""
    return url.split("id=")[1]

def download_pdf(url, output_path, max_retries=3):
    """Download a PDF from URL with retries."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=30, headers={
                'User-Agent': 'Mozilla/5.0 (research paper download)'
            })
            if resp.status_code == 200 and len(resp.content) > 1000:
                with open(output_path, 'wb') as f:
                    f.write(resp.content)
                return True
            else:
                print(f"  Attempt {attempt+1}: status={resp.status_code}, size={len(resp.content)}")
        except Exception as e:
            print(f"  Attempt {attempt+1}: error={e}")
        time.sleep(2 * (attempt + 1))
    return False

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()
    except Exception as e:
        print(f"  Error extracting text: {e}")
        return ""

def main():
    os.makedirs("data/pdfs", exist_ok=True)
    os.makedirs("data/texts", exist_ok=True)
    
    papers = []
    
    for i, url in enumerate(PAPER_URLS):
        paper_id = extract_paper_id(url)
        pdf_path = f"data/pdfs/{paper_id}.pdf"
        text_path = f"data/texts/{paper_id}.txt"
        
        print(f"[{i+1}/50] Processing paper {paper_id}...")
        
        # Download PDF if not already present
        if not os.path.exists(pdf_path):
            print(f"  Downloading...")
            success = download_pdf(url, pdf_path)
            if not success:
                print(f"  FAILED to download {paper_id}")
                continue
            time.sleep(1)  # Be polite
        else:
            print(f"  PDF already exists")
        
        # Extract text if not already done
        if not os.path.exists(text_path) or os.path.getsize(text_path) == 0:
            print(f"  Extracting text...")
            text = extract_text_from_pdf(pdf_path)
            if text:
                with open(text_path, 'w') as f:
                    f.write(text)
                print(f"  Extracted {len(text)} chars")
            else:
                print(f"  FAILED to extract text")
                continue
        else:
            with open(text_path, 'r') as f:
                text = f.read()
            print(f"  Text already extracted ({len(text)} chars)")
        
        papers.append({
            "index": i,
            "paper_id": paper_id,
            "url": url,
            "pdf_path": pdf_path,
            "text_path": text_path,
            "text_length": len(text)
        })
    
    # Save metadata
    with open("data/papers_metadata.json", 'w') as f:
        json.dump(papers, f, indent=2)
    
    print(f"\nSuccessfully processed {len(papers)}/50 papers")
    print(f"Metadata saved to data/papers_metadata.json")

if __name__ == "__main__":
    main()
