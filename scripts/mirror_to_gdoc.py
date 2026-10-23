#!/usr/bin/env python
"""Mirror PAPER.md to the "AI-Enabled Beach Carrying Capacity" Google Doc.

Rerunnable: clears the doc body, then rewrites everything (headings, tables,
lists, images, code blocks) in Times New Roman (fenced code in Courier New).

Usage:
  python mirror_to_gdoc.py [--dry-run]

Auth: uses $LOCALAPPDATA/hermes/google_token.json (documents + drive scopes).
Doc ID: DOC_ID below (the mirror target).
"""
import argparse
import base64
import json
import os
import re
import sys

DOC_ID = '1n3q6fQ4IkA6jE8ouTEbsKCloNAYlp7isZkFlG3a_xLA'
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD_PATH = os.path.join(REPO, 'PAPER.md')

HERMES_HOME = os.path.expandvars(r'$LOCALAPPDATA/hermes')
TOKEN_PATH = os.path.join(HERMES_HOME, 'google_token.json')

MAX_WIDTH_PT = 432.0   # 6 in
MAX_HEIGHT_PT = 384.0  # ~5.3 in
BATCH_REQ_LIMIT = 80
BATCH_BYTE_LIMIT = 700_000


# ---------------------------------------------------------------- markdown --
def parse_blocks(md_text):
    """Parse PAPER.md into a list of (kind, data) blocks."""
    lines = md_text.splitlines()
    blocks = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped == '---':
            i += 1
            continue
        m = re.match(r'^(#{1,6})\s+(.*)$', stripped)
        if m:
            blocks.append(('h%d' % len(m.group(1)), m.group(2).strip()))
            i += 1
            continue
        if stripped.startswith('```'):
            j = i + 1
            code_lines = []
            while j < n and not lines[j].strip().startswith('```'):
                code_lines.append(lines[j])
                j += 1
            blocks.append(('code', code_lines))
            i = j + 1
            continue
        m = re.match(r'^!\[([^\]]*)\]\(([^)]+)\)\s*$', stripped)
        if m:
            blocks.append(('img', (m.group(1).strip(), m.group(2).strip())))
            i += 1
            continue
        if stripped.startswith('|'):
            rows = []
            while i < n and lines[i].strip().startswith('|'):
                row = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-{2,}:?', c) for c in row):
                    rows.append(row)
                i += 1
            if rows:
                blocks.append(('table', rows))
            continue
        m = re.match(r'^\d+\.\s+(.*)$', stripped)
        if m:
            items, i2 = collect_list(lines, i, n, 'num')
            if items:
                blocks.append(('list_num', items))
                i = i2
                continue
        if re.match(r'^[-*]\s+', stripped):
            items, i2 = collect_list(lines, i, n, 'bullet')
            if items:
                blocks.append(('list_bullet', items))
                i = i2
                continue
        # paragraph: accumulate until blank line / next block type
        para = [stripped]
        i += 1
        while i < n:
            s = lines[i].strip()
            if (not s or s == '---' or s.startswith('#') or s.startswith('|')
                    or s.startswith('```') or s.startswith('![')
                    or re.match(r'^\d+\.\s', s) or re.match(r'^[-*]\s', s)):
                break
            para.append(s)
            i += 1
        blocks.append(('para', ' '.join(para)))
    return blocks


def collect_list(lines, i, n, kind):
    """Collect a markdown list starting at line i. kind: 'num' or 'bullet'.
    Numbered items must run 1,2,3,... so wrapped lines like '2026. text' are
    not mistaken for item starts. Returns (items, next_i)."""
    num_pat = re.compile(r'^(\d+)\.\s+(.*)$')
    items = []
    cur = None
    expect = 1
    while i < n:
        s = lines[i].strip()
        if kind == 'num':
            m = num_pat.match(s)
            if m and int(m.group(1)) != expect:
                m = None  # wrapped year etc. — fall through to continuation
        else:
            m = re.match(r'^[-*]\s+(.*)$', s)
        if m:
            if cur is not None:
                items.append(cur)
            cur = m.group(2) if kind == 'num' else m.group(1)
            expect += 1
            i += 1
        elif cur is not None and re.match(r'^\s{2,}\S', lines[i]):
            cur += ' ' + lines[i].strip()
            i += 1
        else:
            break
    if cur is not None:
        items.append(cur)
    return items, i


def render_inline(text):
    """Strip **bold** / *italic* / `code` markers.
    Returns (plain_text, runs) where runs = [(start, end, {'bold'|'italic'})].
    Offsets are relative to plain_text.
    """
    out = []
    runs = []
    i = 0
    n = len(text)
    while i < n:
        if text.startswith('**', i):
            j = text.find('**', i + 2)
            if j == -1:
                j = n
                inner = text[i + 2:]
                bold = False
            else:
                inner = text[i + 2:j]
                bold = True
            s = len(''.join(out))
            out.append(inner)
            if bold:
                runs.append((s, s + len(inner), {'bold': True}))
            i = j + 2 if bold else n
        elif text[i] == '`':
            j = text.find('`', i + 1)
            if j == -1:
                out.append(text[i + 1:])
                i = n
            else:
                out.append(text[i + 1:j])
                i = j + 1
        elif text[i] == '*':
            j = text.find('*', i + 1)
            if j == -1:
                out.append(text[i + 1:])
                i = n
            else:
                s = len(''.join(out))
                inner = text[i + 1:j]
                out.append(inner)
                runs.append((s, s + len(inner), {'italic': True}))
                i = j + 1
        else:
            m = re.search(r'\*\*|\*|`', text[i:])
            if m:
                out.append(text[i:i + m.start()])
                i += m.start()
            else:
                out.append(text[i:])
                i = n
    return ''.join(out), runs


# ------------------------------------------------------------------- images --
def image_size(path):
    with open(path, 'rb') as f:
        data = f.read()
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        w, h = int.from_bytes(data[16:20], 'big'), int.from_bytes(data[20:24], 'big')
        return w, h, 'image/png'
    if data[:2] == b'\xff\xd8':
        i = 2
        while i < len(data) - 9:
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3):
                h, w = int.from_bytes(data[i + 5:i + 7], 'big'), int.from_bytes(data[i + 7:i + 9], 'big')
                return w, h, 'image/jpeg'
            seg = int.from_bytes(data[i + 2:i + 4], 'big')
            i += 2 + seg
    raise ValueError('unsupported image: %s' % path)


def image_object_size(path):
    w, h, _ = image_size(path)
    width = MAX_WIDTH_PT
    height = width * h / w
    if height > MAX_HEIGHT_PT:
        height = MAX_HEIGHT_PT
        width = height * w / h
    return {'width': {'magnitude': round(width, 1), 'unit': 'PT'},
            'height': {'magnitude': round(height, 1), 'unit': 'PT'}}


# ------------------------------------------------------------- request build --
HEADING_MAP = {'h1': 'TITLE', 'h2': 'HEADING_1', 'h3': 'HEADING_2',
               'h4': 'HEADING_3', 'h5': 'HEADING_4', 'h6': 'HEADING_5'}
TNR = {'weightedFontFamily': {'fontFamily': 'Times New Roman', 'weight': 400}}
COURIER = {'weightedFontFamily': {'fontFamily': 'Courier New', 'weight': 400}}


def build_requests(blocks, repo):
    """Deterministically compute all batchUpdate requests.
    Returns (batches, cursor_end, code_ranges) where batches is a list of
    (tag, [requests]); all indices are final-document coordinates."""
    units = []
    code_ranges = []
    cursor = 1  # position of the '\n' of the initial empty paragraph

    def style_runs(base, runs):
        for s, e, st in runs:
            reqs.append({'updateTextStyle': {
                'range': {'startIndex': base + s, 'endIndex': base + e},
                'textStyle': st,
                'fields': ','.join(st.keys())}})

    for kind, data in blocks:
        reqs = []
        if kind.startswith('h'):
            text = data
            plain, runs = render_inline(text)
            L = len(plain) + 1
            reqs.append({'insertText': {'location': {'index': cursor}, 'text': plain + '\n'}})
            reqs.append({'updateParagraphStyle': {
                'range': {'startIndex': cursor, 'endIndex': cursor + L},
                'paragraphStyle': {'namedStyleType': HEADING_MAP[kind]},
                'fields': 'namedStyleType'}})
            style_runs(cursor, runs)
            cursor += L
        elif kind == 'para':
            plain, runs = render_inline(data)
            L = len(plain) + 1
            reqs.append({'insertText': {'location': {'index': cursor}, 'text': plain + '\n'}})
            style_runs(cursor, runs)
            cursor += L
        elif kind == 'img':
            alt, src = data
            path = os.path.join(repo, src)
            b64 = base64.b64encode(open(path, 'rb').read()).decode()
            mime = image_size(path)[2]
            uri = 'data:%s;base64,%s' % (mime, b64)
            size = image_object_size(path)
            # new empty para, image in it, then caption para
            reqs.append({'insertText': {'location': {'index': cursor}, 'text': '\n'}})
            reqs.append({'insertInlineImage': {
                'location': {'index': cursor + 1}, 'uri': uri, 'objectSize': size}})
            reqs.append({'updateParagraphStyle': {
                'range': {'startIndex': cursor + 1, 'endIndex': cursor + 3},
                'paragraphStyle': {'alignment': 'CENTER'}, 'fields': 'alignment'}})
            cap = alt
            cstart = cursor + 2
            reqs.append({'insertText': {'location': {'index': cstart}, 'text': cap + '\n'}})
            reqs.append({'updateTextStyle': {
                'range': {'startIndex': cstart, 'endIndex': cstart + len(cap)},
                'textStyle': {'italic': True}, 'fields': 'italic'}})
            reqs.append({'updateParagraphStyle': {
                'range': {'startIndex': cstart, 'endIndex': cstart + len(cap) + 1},
                'paragraphStyle': {'alignment': 'CENTER'}, 'fields': 'alignment'}})
            cursor = cstart + len(cap) + 1
        elif kind == 'table':
            rows = data
            R = len(rows)
            C = len(rows[0])
            footprint = 2 * R * C + R + 2
            reqs.append({'insertTable': {'location': {'index': cursor},
                                         'rows': R, 'columns': C}})
            off = 0
            for r, row in enumerate(rows):
                for c in range(C):
                    cell = row[c] if c < len(row) else ''
                    plain, runs = render_inline(cell)
                    pos = cursor + 4 + 2 * (r * C + c) + r + off
                    reqs.append({'insertText': {'location': {'index': pos}, 'text': plain}})
                    style_runs(pos, runs)
                    off += len(plain)
            cursor += 1 + footprint
        elif kind in ('list_num', 'list_bullet'):
            items = data
            start = cursor
            for item in items:
                plain, runs = render_inline(item)
                L = len(plain) + 1
                reqs.append({'insertText': {'location': {'index': cursor}, 'text': plain + '\n'}})
                style_runs(cursor, runs)
                cursor += L
            preset = ('NUMBERED_DECIMAL_ALPHA_ROMAN' if kind == 'list_num'
                      else 'BULLET_DISC_CIRCLE_SQUARE')
            reqs.append({'createParagraphBullets': {
                'range': {'startIndex': start, 'endIndex': cursor}, 'bulletPreset': preset}})
        elif kind == 'code':
            text = '\n'.join(data) + '\n'
            L = len(text)
            reqs.append({'insertText': {'location': {'index': cursor}, 'text': text}})
            cursor += L
            code_ranges.append((cursor - L, cursor - 1))
        else:
            raise ValueError('unknown block %r' % kind)
        units.append((kind, reqs))

    # final font pass: Times New Roman everywhere, then Courier over code ranges
    final = []
    final.append(('font-tnr', [{'updateTextStyle': {
        'range': {'startIndex': 1, 'endIndex': cursor + 1},
        'textStyle': TNR, 'fields': 'weightedFontFamily'}}]))
    for s, e in code_ranges:
        final.append(('font-code', [{'updateTextStyle': {
            'range': {'startIndex': s, 'endIndex': e + 1},
            'textStyle': COURIER, 'fields': 'weightedFontFamily'}}]))
    units.extend(final)

    # pack units into batches (each image unit gets its own batch)
    batches = []
    cur = []
    cur_bytes = 0
    for tag, reqs in units:
        size = sum(len(json.dumps(r)) for r in reqs)
        is_img = any('insertInlineImage' in r for r in reqs)
        if is_img:
            if cur:
                batches.append(cur)
                cur, cur_bytes = [], 0
            batches.append(reqs)
            continue
        if cur and (len(cur) + len(reqs) > BATCH_REQ_LIMIT or cur_bytes + size > BATCH_BYTE_LIMIT):
            batches.append(cur)
            cur, cur_bytes = [], 0
        cur.extend(reqs)
        cur_bytes += size
    if cur:
        batches.append(cur)
    return batches, cursor, code_ranges


# ---------------------------------------------------------------------- api --
def make_services():
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    tok = json.load(open(TOKEN_PATH))
    creds = Credentials(None, refresh_token=tok['refresh_token'], token_uri=tok['token_uri'],
                        client_id=tok['client_id'], client_secret=tok['client_secret'],
                        scopes=tok['scopes'])
    docs = build('docs', 'v1', credentials=creds, cache_discovery=False).documents()
    drive = build('drive', 'v3', credentials=creds, cache_discovery=False).files()
    return docs, drive


def body_cursor(docs):
    d = docs.get(documentId=DOC_ID).execute()
    paras = [b for b in d['body']['content'] if 'paragraph' in b]
    return max(b['endIndex'] for b in paras) - 1


def clear_doc(docs):
    c = body_cursor(docs)
    if c > 1:
        docs.batchUpdate(documentId=DOC_ID, body={'requests': [
            {'deleteContentRange': {'range': {'startIndex': 1, 'endIndex': c},
                                    'type': 'RESTART_NUMBERING'}}]}).execute()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    md = open(MD_PATH, encoding='utf-8').read()
    blocks = parse_blocks(md)
    batches, cursor_end, code_ranges = build_requests(blocks, REPO)

    n_img = sum(1 for k, _ in blocks if k == 'img')
    n_tbl = sum(1 for k, _ in blocks if k == 'table')
    print('blocks: %d total, %d images, %d tables, %d code blocks'
          % (len(blocks), n_img, n_tbl, sum(1 for k, _ in blocks if k == 'code')))
    print('batches: %d, requests: %d, final cursor: %d'
          % (len(batches), sum(len(b) for b in batches), cursor_end))

    if args.dry_run:
        for b in batches:
            print('  batch(%d): %s' % (len(b), sorted({next(iter(r)) for r in b})))
        return

    docs, drive = make_services()
    title = blocks[0][1] if blocks[0][0] == 'h1' else 'Beach Carrying Capacity'
    clear_doc(docs)
    cursor = 1
    for bi, batch in enumerate(batches, 1):
        docs.batchUpdate(documentId=DOC_ID, body={'requests': batch}).execute()
        actual = body_cursor(docs)
        # expected cursor after this batch = advance cursor by the batch's text
        # simpler: trust and report; assert monotonic + final check below
        cursor = actual
        print('batch %2d/%d done (%3d reqs), body cursor = %d'
              % (bi, len(batches), len(batch), actual))
    final = body_cursor(docs)
    print('final body cursor: %d (expected %d) %s'
          % (final, cursor_end, 'OK' if final == cursor_end else 'MISMATCH!'))

    d = docs.get(documentId=DOC_ID).execute()
    n_inline = len(d.get('inlineObjects', {}))
    print('inline images in doc: %d' % n_inline)

    drive.patch(fileId=DOC_ID, body={'name': title}, fields='name').execute()
    print('drive title set: %s' % title)
    print('DONE')


if __name__ == '__main__':
    main()
