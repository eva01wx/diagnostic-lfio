"""Render the final manuscript for visual review and record structural checks."""
import hashlib
import argparse
import json
import math
from pathlib import Path
import re
import pymupdf
from PIL import Image,ImageDraw


def run():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper', type=Path, default=Path('LFIO_Negation_Paper.pdf'))
    parser.add_argument('--output', type=Path, default=Path('docs/research/kbs_2026-09-24/manuscript_audit.json'))
    parser.add_argument('--render-dir', type=Path, default=Path('results/tex_build/cas_20260924'))
    args=parser.parse_args()
    paper=args.paper
    render=args.render_dir
    render.mkdir(parents=True,exist_ok=True)
    doc=pymupdf.open(paper)
    texts=[];blocks=[];thumbs=[];page_map={}
    needles=['Fidelity under explicit','Primary eight-condition results','Matched-budget',
             'Complexity-counting sensitivity','Computation per dataset',
             'Training-objective gaps','Extension details and reproduction',
             'Generated tasks:', 'Equal validation opportunities', 'Native numerical teachers',
             'Concrete repair trajectories', 'Retrospective teacher-quality']
    for index,page in enumerate(doc):
        text=page.get_text();texts.append(text)
        for needle in needles:
            if needle in text:
                page_map.setdefault(needle,[]).append(index+1)
        for block in page.get_text('blocks'):
            if block[0]<-1 or block[2]>page.rect.width+1:
                blocks.append({'page':index+1,'bounds':list(block[:4])})
        pix=page.get_pixmap(matrix=pymupdf.Matrix(1.5,1.5),alpha=False)
        target=render/f'page-{index+1:02d}.png';pix.save(target)
        thumb=Image.open(target).convert('RGB');thumb.thumbnail((235,330))
        tile=Image.new('RGB',(245,355),'white')
        tile.paste(thumb,((245-thumb.width)//2,20))
        ImageDraw.Draw(tile).text((8,4),f'Page {index+1}',fill='black')
        thumbs.append(tile)
    cols=5;sheet=Image.new('RGB',(cols*245,math.ceil(len(thumbs)/cols)*355),'#cccccc')
    for i,thumb in enumerate(thumbs):
        sheet.paste(thumb,((i%cols)*245,(i//cols)*355))
    sheet.save(render/'contact-sheet.png')
    log=paper.with_suffix('.log').read_text(errors='replace')
    warnings=[line.strip() for line in log.splitlines() if 'Overfull' in line or 'LaTeX Warning:' in line]
    underfull=[line.strip() for line in log.splitlines() if 'Underfull' in line]
    package_warnings=[line.strip() for line in log.splitlines() if re.search(r'Package .* Warning:',line)]
    bibtex_log=paper.with_suffix('.blg')
    bibtex_warnings=([line.strip() for line in bibtex_log.read_text(errors='replace').splitlines()
                     if 'Warning--' in line or 'error message' in line] if bibtex_log.exists() else [])
    result={'pages':len(doc),'unresolved_reference_markers':'??' in '\n'.join(texts),
            'warnings':warnings,'package_warnings':package_warnings,'bibtex_warnings':bibtex_warnings,
            'underfull_boxes':underfull,'outside_page_text_blocks':blocks,'review_page_map':page_map,
            'pdf_sha256':hashlib.sha256(paper.read_bytes()).hexdigest(),
            'tex_sha256':hashlib.sha256(paper.with_suffix('.tex').read_bytes()).hexdigest(),
            'body_source_sha256':hashlib.sha256(Path('LFIO_Negation_Paper.tex').read_bytes()).hexdigest(),
            'visual_review_complete':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    run()
