#!/usr/bin/env bash
# Download and unpack the thermal benchmarks of the RGB -> thermal extension, then list what they contain.
#
#   bash tools/download_thermal.sh flir <url>            the link given by the FLIR ADAS v2 form (the download is gated)
#   bash tools/download_thermal.sh flir <file.zip>       an archive already on disk
#   bash tools/download_thermal.sh m3fd                  the Google Drive folder of the TarDAL README (needs gdown or uvx)
#   bash tools/download_thermal.sh m3fd <drive url>      one Drive file or folder (if the README folder holds more than M3FD)
#   bash tools/download_thermal.sh m3fd <archive|dir>    an archive or folder already on disk (e.g. from Baidu Yun)
#
# Everything goes to $DATA/<dataset>/ (DATA defaults to <repo>/datasets): archives in download/, contents in raw/.
# Nothing existing is overwritten: downloads resume, unpacking skips existing files. Archives are kept; the script
# prints the command to delete them once the conversion is checked. At the end it writes <dataset>/DOWNLOAD.txt
# (source, sizes, sha256) and <dataset>/listing.txt (FLIR: coco.json files with images and boxes per category;
# M3FD: image / label counts, class ids, image sizes, pairing). Long downloads: run under nohup or tmux.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA=${DATA:-$REPO/datasets}
PY=${PY:-}
for c in python3 python; do   # the first one that runs (python3 can be a stub that only prints a message)
    [[ -z $PY ]] && command -v $c >/dev/null 2>&1 && $c -c "" >/dev/null 2>&1 && PY=$c
done
M3FD_DRIVE=https://drive.google.com/drive/folders/1H-oO7bgRuVFYDcMGvxstT1nmy0WF_Y_6   # TarDAL README, "M3FD Dataset"

die() { echo "error: $*" >&2; exit 1; }
[[ -n $PY ]] || die "python not found (set PY=...)"

fetch() {  # fetch <url> <file>: resumable
    mkdir -p "$(dirname "$2")"
    echo "+ download -> $2"
    if command -v wget >/dev/null 2>&1; then wget -c --progress=dot:giga -O "$2" "$1"
    else curl -fL -C - -o "$2" "$1"; fi
}

unpack() {  # unpack <archive> <dir>: existing files are skipped
    local a=$1 dest=$2
    mkdir -p "$dest"
    echo "+ unpack $a -> $dest"
    case ${a,,} in
        *.zip) unzip -q -n "$a" -d "$dest" ;;
        *.tar|*.tar.gz|*.tgz|*.tar.xz|*.tar.bz2) tar -xf "$a" -C "$dest" --skip-old-files ;;
        *.7z|*.rar)
            if command -v 7z >/dev/null 2>&1; then 7z x -aos -o"$dest" "$a" >/dev/null
            elif command -v 7za >/dev/null 2>&1; then 7za x -aos -o"$dest" "$a" >/dev/null
            elif [[ ${a,,} == *.rar ]] && command -v unrar >/dev/null 2>&1; then unrar x -o- "$a" "$dest/" >/dev/null
            else die "no 7z / unrar here to unpack $a: unpack it elsewhere and pass the folder"; fi ;;
        *) die "unknown archive type: $a" ;;
    esac
}

archives() {  # archives <dir>: the archives directly inside a folder
    find "$1" -maxdepth 1 -type f \( -iname '*.zip' -o -iname '*.rar' -o -iname '*.7z' -o -iname '*.tar' \
        -o -iname '*.tgz' -o -iname '*.tar.gz' -o -iname '*.tar.xz' -o -iname '*.tar.bz2' \) | sort
}

record() {  # record <dataset dir> <source> <archive>...: DOWNLOAD.txt (no query string: links may carry tokens)
    local root=$1 out=$1/DOWNLOAD.txt src=${2%%\?*}; shift 2
    {
        echo "date: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "source: $src"
        for f in "$@"; do
            [[ -f $f ]] && echo "archive: $(basename "$f")  bytes $(stat -c %s "$f" 2>/dev/null || wc -c < "$f")  sha256 $(sha256sum "$f" | cut -d' ' -f1)"
        done
        echo "unpacked: $(du -sh "$root/raw" 2>/dev/null | cut -f1)"
    } | tee "$out"
}

gdown_cmd() {
    if command -v gdown >/dev/null 2>&1; then echo gdown
    elif command -v uvx >/dev/null 2>&1; then echo "uvx gdown"
    else die "M3FD from Google Drive needs gdown: install uv (uvx) or 'pip install --user gdown'"; fi
}

list_flir() {
    "$PY" - "$1" <<'EOF'
import collections, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
files = sorted(root.rglob("coco.json"))
if not files:
    sys.exit(f"no coco.json under {root}")
for f in files:
    c = json.loads(f.read_text())
    names = {k["id"]: k["name"] for k in c["categories"]}
    n = collections.Counter(names.get(a["category_id"], a["category_id"]) for a in c["annotations"])
    sample = [i["file_name"] for i in c["images"][:5]]
    found = sum((f.parent / s).exists() for s in sample)
    print(f"{f.parent.relative_to(root.parent)}: {len(c['images'])} images, {len(c['annotations'])} boxes; "
          f"file_name e.g. {sample[0] if sample else '-'} ({found}/{len(sample)} found next to coco.json)")
    print("    " + ", ".join(f"{k} {v}" for k, v in n.most_common()))
    print(f"    categories ({len(names)}): " + ", ".join(f"{i}={nm}" for i, nm in sorted(names.items())))
EOF
}

list_m3fd() {
    "$PY" - "$1" <<'EOF'
import collections, pathlib, struct, sys
root = pathlib.Path(sys.argv[1])
IMG = {".bmp", ".png", ".jpg", ".jpeg"}

def size(p):
    with p.open("rb") as f:
        h = f.read(26)
    if h[:2] == b"BM":
        return struct.unpack("<ii", h[18:26])
    if h[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", h[16:24])
    return None

dirs = collections.defaultdict(list)
for p in root.rglob("*"):
    if p.is_file():
        dirs[p.parent].append(p)
stems = {}
for d, fs in sorted(dirs.items()):
    exts = collections.Counter(p.suffix.lower() for p in fs)
    print(f"{d.relative_to(root.parent)}: {len(fs)} files " + ", ".join(f"{e or '-'} {n}" for e, n in exts.most_common()))
    imgs = [p for p in fs if p.suffix.lower() in IMG]
    if imgs:
        sizes = collections.Counter(size(p) for p in imgs)
        print("    sizes: " + ", ".join(f"{s[0]}x{s[1]} {n}" if s else f"unknown {n}" for s, n in sizes.most_common(5)))
        stems[d] = {p.stem for p in imgs}
    txts = [p for p in fs if p.suffix.lower() == ".txt"]
    if len(txts) > 10:  # YOLO labels: one line "class cx cy w h" per box
        ids, boxes, bad = collections.Counter(), 0, 0
        for p in txts:
            for line in p.read_text().split("\n"):
                parts = line.split()
                if not parts:
                    continue
                if len(parts) != 5:
                    bad += 1
                    continue
                ids[parts[0]] += 1
                boxes += 1
        print(f"    YOLO labels: {boxes} boxes, class ids " + ", ".join(f"{k}: {v}" for k, v in sorted(ids.items()))
              + (f"; {bad} lines not in 'class cx cy w h' form" if bad else ""))
        stems[d] = {p.stem for p in txts}
    elif txts:
        for p in txts:
            print(f"    {p.name}: {sum(1 for l in p.read_text().split(chr(10)) if l.strip())} lines")
if len(stems) >= 2:
    ref_dir, ref = max(stems.items(), key=lambda kv: len(kv[1]))
    for d, s in stems.items():
        if d != ref_dir:
            print(f"pairing {d.name} vs {ref_dir.name}: {len(s & ref)} common, {len(s - ref)} only in {d.name}, "
                  f"{len(ref - s)} only in {ref_dir.name}")
EOF
}

flir() {
    local src=${1:-} root=$DATA/flir archive
    [[ -n $src ]] || die "flir needs the link given by the FLIR ADAS v2 form, or a downloaded zip"
    if [[ -f $src ]]; then archive=$src; else archive=$root/download/FLIR_ADAS_v2.zip; fetch "$src" "$archive"; fi
    [[ $(head -c 2 "$archive") == PK ]] || die "$archive is not a zip (an expired link returns a web page): get a new link"
    unpack "$archive" "$root/raw"
    record "$root" "$src" "$archive"
    list_flir "$root/raw" | tee "$root/listing.txt"
    echo "once the conversion is checked: rm $archive"
}

m3fd() {
    local src=${1:-} root=$DATA/m3fd
    local -a found
    if [[ -z $src || $src == http* ]]; then
        local url=${src:-$M3FD_DRIVE} gd
        gd=$(gdown_cmd)
        mkdir -p "$root/download"
        echo "+ $gd -> $root/download ($url)"
        if [[ $url == */folders/* ]]; then $gd --folder "$url" -O "$root/download" --remaining-ok
        else (cd "$root/download" && $gd --fuzzy "$url"); fi
        src=$root/download
    fi
    if [[ -d $src ]]; then
        mapfile -t found < <(archives "$src")
        if (( ${#found[@]} )); then
            for a in "${found[@]}"; do unpack "$a" "$root/raw"; done
        else
            echo "no archive in $src: listing it as the unpacked data"
            list_m3fd "$src" | tee "$root/listing.txt"
            return
        fi
    else
        found=("$src")
        unpack "$src" "$root/raw"
    fi
    record "$root" "${1:-$M3FD_DRIVE}" "${found[@]}"
    list_m3fd "$root/raw" | tee "$root/listing.txt"
    echo "once the conversion is checked: rm ${found[*]}"
}

case ${1:-} in
    flir|m3fd)
        echo "data folder: $DATA ($(df -h "$DATA" 2>/dev/null | awk 'NR==2 {print $4 " free on " $6}'))"
        "$@" ;;
    *) sed -n '2,15p' "${BASH_SOURCE[0]}"; exit 2 ;;
esac
